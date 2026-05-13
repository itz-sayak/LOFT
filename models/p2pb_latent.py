import torch
import torch.nn as nn
import torch.nn.functional as F
from models.p2pb import P2PB, space_indices, align_to_bnc
from models.autoencoder import GeomTokenAE
from models.freq_encoding_transformer import FreqEncodingTransformer
from loguru import logger


class LatentP2PB(P2PB):
    """Latent-space P2P-Bridge with frozen AE conditioning + FreqEncodingTransformer."""

    def __init__(self, cfg, model=None):
        super().__init__(cfg, model)

        # ---- Load frozen AE ----
        self.ae = GeomTokenAE(
            in_dim=getattr(cfg.model, "in_dim", 3),
            feat_dim=getattr(cfg.model, "feat_dim", 512),
            use_dino=getattr(cfg.model, "use_dino", False),
            use_offset_attn=getattr(cfg.model, "use_offset_attn", True),
        )

        ae_ckpt_path = getattr(cfg.model, "ae_ckpt", "")
        try:
            ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=True)
            if "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            self.ae.load_state_dict(ckpt, strict=False)
            print(f"  AE checkpoint loaded  : {ae_ckpt_path}")
        except Exception as e:
            logger.warning(f"Failed to load AE checkpoint from {ae_ckpt_path}: {e}")

        for p in self.ae.parameters():
            p.requires_grad = False
        self.ae.eval()
        logger.info("Frozen AE initialized for latent conditioning.")

        # ---- Trainable Frequency Encoder Transformer ----
        latent_dim = getattr(cfg.model, "feat_dim", 512)
        self.freq_transformer = FreqEncodingTransformer(
            latent_dim=latent_dim,
            num_tokens=getattr(cfg.model, "ae_num_tokens", 64),
            num_freqs=getattr(cfg.model, "freq_num_freqs", 6),
            num_layers=getattr(cfg.model, "freq_num_layers", 4),
            num_heads=getattr(cfg.model, "freq_num_heads", 8),
            ff_mult=getattr(cfg.model, "freq_ff_mult", 4),
            dropout=getattr(cfg.model, "freq_dropout", 0.1),
            out_dim=getattr(cfg.model, "freq_out_dim", latent_dim),
            learnable_freq=getattr(cfg.model, "freq_learnable", True),
            time_embed_dim=getattr(cfg.model, "time_embed_dim", 64),
        )
        logger.info("FreqEncodingTransformer initialized (trainable.)")

    # ------------------------------------------------------------
    def _to_b3n(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure xyz-only [B,3,N] layout regardless of dataloader shape."""
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,N] or [B,N,C], got {x.shape}")
        B, D1, D2 = x.shape

        if D1 == 3:
            return x
        if D2 == 3:
            return x.transpose(1, 2).contiguous()
        if D1 > 3 and D2 != 3:
            return x[:, :3, :].contiguous()
        if D2 > 3:
            return x.transpose(1, 2)[:, :3, :].contiguous()
        raise RuntimeError(f"Cannot determine xyz layout from {x.shape}")

    # ------------------------------------------------------------
    def forward(self, x0, x1, x_cond=None):
        device = x0.device
        B = x0.shape[0]

        steps = torch.randint(0, self.timesteps, (B,), device=device)
        xt = self.q_sample(steps, x0, x1)
        gt = self.compute_gt(steps, x0, xt)

        # q_sample/compute_gt run align_to_bnc → output is BNC [B, N, 3].
        # PVCNN2Unet expects BCN [B, 3, N] — transpose if needed.
        if xt.ndim == 3 and xt.shape[-1] == 3 and xt.shape[1] != 3:
            xt = xt.transpose(1, 2).contiguous()   # [B, 3, N]
        if gt.ndim == 3 and gt.shape[-1] == 3 and gt.shape[1] != 3:
            gt = gt.transpose(1, 2).contiguous()   # [B, 3, N]

        # ---- Encode AE latent (frozen) ----
        with torch.no_grad():
            xyz = self._to_b3n(x1[:, :3, :])
            ae_latent_raw = self.ae.encode(xyz)
            xyz_clean = self._to_b3n(x0[:, :3, :])
            ae_latent_clean = self.ae.encode(xyz_clean)

        noise_t = self.noise_levels[steps].detach()
        latent = self.freq_transformer(ae_latent_raw, t=noise_t)
        cond_tokens = latent

        # CFG dropout
        cond_drop_prob = self.cfg.diffusion.get("cond_drop_prob", 0.0)
        if self.training and cond_drop_prob > 0.0 and torch.rand(1, device=device).item() < cond_drop_prob:
            cond_tokens = torch.zeros_like(cond_tokens)

        # ---- Diffusion prediction ----
        pred_out = self.model(xt, t=steps, x_cond=cond_tokens)
        pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        if pred.shape != gt.shape:
            gt = gt.transpose(1, 2).contiguous()

        # ---- Loss computation ----
        per_sample_loss = self.calculate_loss(pred, gt)  # [B]
        loss = per_sample_loss.mean()

        # ---- Latent consistency loss ----
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
        """Override parent sample() to run AE conditioning before denoising."""
        assert x_start is not None, "LatentP2PB.sample() requires x_start"

        # ---- Encode AE latent (frozen, computed once) ----
        xyz = self._to_b3n(x_start)                     # [B, 3, N] BCN
        B, _, N = xyz.shape
        ae_latent = self.ae.encode(xyz)                  # [B, M, 512]
        self.freq_transformer.eval()

        # ---- DDPM step schedule ----
        sampling_steps = self.cfg.diffusion.sampling_timesteps if steps is None else steps
        assert 0 < sampling_steps < self.timesteps == len(self.betas)

        step_indices = space_indices(self.timesteps, sampling_steps + 1)
        lc = min(len(step_indices) - 1, log_count)
        log_steps = [step_indices[i] for i in space_indices(len(step_indices) - 1, lc)]
        assert log_steps[0] == 0

        if verbose:
            logger.info(f"[LatentP2PB Sampling] T={self.timesteps}, sampling_steps={sampling_steps}!")

        self.model.eval()

        def pred_x0_fn(xt, step, x1, x_cond=None):
            xt_bnc = align_to_bnc(xt)
            xt_bcn = xt_bnc.transpose(1, 2).contiguous()

            step_t = torch.full((xt_bcn.shape[0],), step,
                                device=xt_bcn.device, dtype=torch.long)
            noise_levels = self.noise_levels[step_t].detach()

            latent = self.freq_transformer(ae_latent, t=noise_levels)  # [B, M, 512]
            x_cond_enc = latent

            model_fn = self.ema if (use_ema and self.ema is not None) else self.model
            out = model_fn(xt_bcn, noise_levels, x_cond=x_cond_enc)
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


