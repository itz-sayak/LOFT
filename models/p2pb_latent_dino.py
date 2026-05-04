"""
LatentP2PBDino — Latent P2P-Bridge with *both* AE latent conditioning
(via FreqEncodingTransformer + WriteAttention) **and** per-point DINO
feature conditioning (concatenated as extra input channels).

This is the indoor-scene variant described in the P2P-Bridge paper:
    - RGB+DINO features serve as per-point visual guidance  →  extra input channels
    - Frozen AE latent + FreqTransformer serve as global structure guidance  →  cross-attn injection

Usage (in config):
    model:
      model_type: latentp2pb_dino
      extra_feature_channels: 384   # DINO dim (or 387 if RGB+DINO)
      ae_ckpt: /path/to/ae_epoch_675.pth
      ...

Training:
    The dataloader returns batch["noisy_features"] (DINO, [B, 384, N]) which
    flows through get_data_batch() → data_batch["x_cond"] → model.forward(x_cond=...).

Inference:
    denoise_room.py already passes patch_dino as x_cond to model.sample().
"""

import torch
import torch.nn.functional as F

from loguru import logger
from models.p2pb import align_to_bnc
from models.p2pb_latent import LatentP2PB


class LatentP2PBDino(LatentP2PB):
    """Extends LatentP2PB to also condition on per-point DINO features.

    The DINO features are concatenated with the noisy xyz coords as extra
    input channels to the PVCNN backbone, while the AE latent tokens are
    injected via the existing WriteAttention mechanism.

    Args:
        cfg:   OmegaConf DictConfig — same fields as LatentP2PB, plus
               cfg.model.PVD.extra_feature_channels = 384 (or 387 RGB+DINO).
        model: Instantiated PVCNN2Unet backbone (passed in from model_loader).
    """

    def forward(self, x0, x1, x_cond=None):
        """Forward pass with optional per-point DINO conditioning.

        Args:
            x0:     Clean points  [B, 3, N]  (BCN) or [B, N, 3].
            x1:     Noisy points  [B, 3, N]  (BCN) or [B, N, 3].
            x_cond: DINO features [B, D, N]  (BCN, D=384) or None.
                    When None the model falls back to AE-only conditioning,
                    behaving exactly like LatentP2PB.
        """
        device = x0.device
        B = x0.shape[0]

        steps = torch.randint(0, self.timesteps, (B,), device=device)
        xt = self.q_sample(steps, x0, x1)
        gt = self.compute_gt(steps, x0, xt)

        # Ensure BCN layout for PVCNN
        if xt.ndim == 3 and xt.shape[-1] == 3 and xt.shape[1] != 3:
            xt = xt.transpose(1, 2).contiguous()   # [B, 3, N]
        if gt.ndim == 3 and gt.shape[-1] == 3 and gt.shape[1] != 3:
            gt = gt.transpose(1, 2).contiguous()

        # ---- AE latent (frozen) → FreqTransformer cond tokens ----
        with torch.no_grad():
            xyz_noisy = self._to_b3n(x1[:, :3, :])
            ae_latent_raw = self.ae.encode(xyz_noisy)
            xyz_clean = self._to_b3n(x0[:, :3, :])
            ae_latent_clean = self.ae.encode(xyz_clean)

        noise_t = self.noise_levels[steps].detach()
        latent = self.freq_transformer(ae_latent_raw, t=noise_t)
        cond_tokens = latent   # [B, M, 512] — injected via WriteAttention

        # CFG dropout on AE latent tokens
        cond_drop_prob = self.cfg.diffusion.get("cond_drop_prob", 0.0)
        if self.training and cond_drop_prob > 0.0 and torch.rand(1, device=device).item() < cond_drop_prob:
            cond_tokens = torch.zeros_like(cond_tokens)

        # ---- Concatenate DINO features as extra input channels ----
        xt_input = xt  # default: [B, 3, N]
        if x_cond is not None:
            # x_cond layout: ensure BCN [B, D, N]
            if x_cond.ndim == 3 and x_cond.shape[-1] != x_cond.shape[1]:
                if x_cond.shape[1] == xt.shape[2]:    # [B, N, D] → transpose
                    x_cond = x_cond.transpose(1, 2).contiguous()
            xt_input = torch.cat([xt, x_cond], dim=1)   # [B, 3+D, N]

        # ---- Diffusion prediction ----
        pred_out = self.model(xt_input, t=steps, x_cond=cond_tokens)
        pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        if pred.shape != gt.shape:
            gt = gt.transpose(1, 2).contiguous()

        # ---- Diffusion loss ----
        per_sample_loss = self.calculate_loss(pred, gt)
        loss = per_sample_loss.mean()

        # ---- Latent consistency auxiliary loss ----
        latent_loss_w = float(self.cfg.diffusion.get("latent_loss_weight", 0.0))
        if self.training and latent_loss_w > 0.0:
            lat_pred   = F.normalize(latent.mean(dim=1), dim=-1)
            lat_target = F.normalize(ae_latent_clean.mean(dim=1), dim=-1)
            latent_loss = (1.0 - (lat_pred * lat_target).sum(dim=-1)).mean()
            loss = loss + latent_loss_w * latent_loss

        return loss, cond_tokens

    # ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x_cond=None, x_start=None, clip=False, use_ema=False,
               verbose=True, log_count=10, steps=None):
        """Inference with AE latent + optional per-point DINO conditioning.

        Args:
            x_start: Noisy input [B, 3, N] (BCN).
            x_cond:  DINO features [B, D, N] (BCN) or None.
        """
        from models.p2pb import space_indices

        assert x_start is not None, "LatentP2PBDino.sample() requires x_start"

        # ---- Encode AE latent once (frozen) ----
        xyz = self._to_b3n(x_start)
        ae_latent = self.ae.encode(xyz)
        self.freq_transformer.eval()

        # ---- Ensure x_cond (DINO) is BCN [B, D, N] if provided ----
        dino_feats = None
        if x_cond is not None:
            if x_cond.ndim != 3:
                raise ValueError(f"x_cond must be rank-3, got shape={tuple(x_cond.shape)}")
            n_points = xyz.shape[2]
            if x_cond.shape[2] == n_points:
                # Already BCN: [B, D, N]
                dino_feats = x_cond.contiguous()
            elif x_cond.shape[1] == n_points:
                # BNC: [B, N, D] -> BCN
                dino_feats = x_cond.transpose(1, 2).contiguous()
            else:
                raise ValueError(
                    f"x_cond has incompatible shape={tuple(x_cond.shape)} for N={n_points}; "
                    "expected [B, D, N] or [B, N, D]"
                )

        # ---- DDPM step schedule ----
        sampling_steps = self.cfg.diffusion.sampling_timesteps if steps is None else steps
        assert 0 < sampling_steps < self.timesteps == len(self.betas)

        step_indices = space_indices(self.timesteps, sampling_steps + 1)
        lc = min(len(step_indices) - 1, log_count)
        log_steps = [step_indices[i] for i in space_indices(len(step_indices) - 1, lc)]
        assert log_steps[0] == 0

        if verbose:
            logger.info(f"[LatentP2PBDino Sampling] T={self.timesteps}, steps={sampling_steps}, "
                        f"dino={'yes' if dino_feats is not None else 'no'}")

        self.model.eval()

        def pred_x0_fn(xt, step, x1, x_cond=None):  # x_cond arg unused here; we close over dino_feats
            xt_bnc = align_to_bnc(xt)
            xt_bcn = xt_bnc.transpose(1, 2).contiguous()   # [B, 3, N]

            step_t = torch.full((xt_bcn.shape[0],), step,
                                device=xt_bcn.device, dtype=torch.long)
            noise_levels = self.noise_levels[step_t].detach()

            # Concatenate DINO as extra input channels
            xt_input = xt_bcn
            if dino_feats is not None:
                xt_input = torch.cat([xt_bcn, dino_feats], dim=1)   # [B, 3+D, N]

            latent = self.freq_transformer(ae_latent, t=noise_levels)   # [B, M, 512]

            model_fn = self.ema if (use_ema and self.ema is not None) else self.model
            out = model_fn(xt_input, noise_levels, x_cond=latent)
            if isinstance(out, tuple):
                out = out[0]
            out = align_to_bnc(out)

            return self.compute_pred_x0_from_eps(step_t, xt_bnc, out, clip_denoise=clip)

        xs, pred_x0 = self.sample_ddpm(
            step_indices, pred_x0_fn, align_to_bnc(x_start),
            x_cond=None, log_steps=log_steps, verbose=verbose,
        )

        self.model.train()
        self.freq_transformer.train()
        return {"x_chain": xs, "x_pred": xs[:, 0, ...], "x_start": x_start}
