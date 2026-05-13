"""
OT-CFM replacement for the Diffusion Schrodinger Bridge in P2P-Bridge.

Replaces the DSB bridge formulation with Optimal Transport Conditional Flow
Matching (OT-CFM), leaving the backbone (modified PVCNN) completely unchanged.

Mathematical basis:
  - Point-level OT coupling: for each (x0[i], x1[i]) pair, solve an N×N
    optimal transport to reorder x1's points to match x0's ordering.
    This straightens interpolation paths without corrupting object-level
    pairing.  Uses exact Hungarian algorithm parallelized across the batch.
  - Linear interpolation between OT-reordered pairs with sigma_min noise
  - Velocity MSE loss targeting the constant field (x1_reordered - x0)
  - Euler ODE integration at inference with configurable NFE

OT coupling is a training-only operation.
The backbone is passed in as a constructor argument and is never modified.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from ema_pytorch import EMA
from loguru import logger
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from models.train_utils import DiffusionModel


class OTFlowBridge(DiffusionModel):
    """
    Replaces the Diffusion Schrodinger Bridge in P2P-Bridge with Optimal
    Transport Conditional Flow Matching (OT-CFM).

    The backbone (modified PVCNN) is passed in and left unchanged.
    OT coupling is a training-only operation.
    """

    def __init__(
        self,
        cfg: Dict,
        model: nn.Module,
    ) -> None:
        super().__init__()
        device = cfg.gpu if cfg.gpu is not None else torch.device("cuda")
        self.device = device
        self.cfg = cfg

        # backbone — never modified
        self.model = model.to(device)
        self.ema: Optional[EMA] = EMA(self.model, beta=0.999) if cfg.model.ema else None

        # OT-CFM hyperparameters
        self.sigma_min: float = 1e-4
        self.ot_reg: float = cfg.diffusion.get("ot_reg", 0.14)

        # Scale factor: map t ∈ [0, 1] to the range expected by the
        # backbone's sinusoidal timestep embedding (~0–1000).
        self.timestep_scale: float = 1000.0

        # Reconstruction loss used by evaluation.py (model.loss(pred, gt))
        self._eval_loss_fn = nn.MSELoss()

        # Thread pool for parallel Hungarian (persists for model lifetime)
        self._ot_pool = ThreadPoolExecutor(max_workers=8)

        # OT coupling frequency: 1 = every step (no caching). Exposed so
        # callers can inspect/log this without reading source.
        self.ot_update_freq: int = 1

        # Per-component timing: prints for the first N training steps then
        # silences itself. Set to 0 to disable.
        self._timing_countdown: int = 5
        self._last_ot_ms: float = 0.0

    # ------------------------------------------------------------------
    # Reconstruction loss (used by evaluation.py)
    # ------------------------------------------------------------------
    def loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred = pred.to(self.device)
        gt = gt.to(self.device)
        return self._eval_loss_fn(pred, gt)

    # ------------------------------------------------------------------
    # Point-level OT coupling (training only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ot_coupling(
        self, x0: Tensor, x1: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Point-level OT: reorder x1's points within each sample to minimize
        transport cost to x0.  Object-level pairing is preserved — x0[i] is
        always paired with x1[i]. Only the point ordering within x1[i] changes.

        Uses exact linear assignment (Hungarian algorithm) parallelized across
        the batch via a persistent thread pool.  Cost matrices are computed on
        GPU, transferred to CPU for scipy, and the reordered result is
        assembled back on GPU.

        Args:
            x0: Noisy patches, shape (B, 3, N).
            x1: Clean patches, shape (B, 3, N).

        Returns:
            Tuple of (x0, x1_reordered), each shape (B, 3, N).
            x0 is returned unchanged; x1 has its point axis permuted.
        """
        import time as _time

        B, C, N = x0.shape
        _t0 = _time.perf_counter()

        # Compute N×N cost matrices on GPU: (B, N, N)
        pts0 = x0.detach().transpose(1, 2).contiguous()  # (B, N, 3)
        pts1 = x1.detach().transpose(1, 2).contiguous()  # (B, N, 3)
        cost_gpu = torch.cdist(pts0, pts1, p=2).pow(2)

        # Transfer to CPU for scipy
        cost_cpu = cost_gpu.cpu().numpy().astype(np.float64)

        def _solve(i: int) -> np.ndarray:
            try:
                _, col = linear_sum_assignment(cost_cpu[i])
                return col
            except Exception as exc:  # degenerate cost matrix
                logger.warning(
                    "ot_coupling: linear_sum_assignment failed for sample {} "
                    "({}); falling back to identity assignment.",
                    i, exc,
                )
                return np.arange(N, dtype=np.int64)

        # Parallel Hungarian across batch samples
        col_arrays = list(self._ot_pool.map(_solve, range(B)))

        # Assemble reordered x1 on GPU
        col_idx = torch.from_numpy(np.stack(col_arrays)).to(
            device=x1.device, dtype=torch.long
        )  # (B, N)
        col_idx_expanded = col_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, N)
        x1_reordered = torch.gather(x1, dim=2, index=col_idx_expanded)

        self._last_ot_ms = (_time.perf_counter() - _t0) * 1000.0
        return x0, x1_reordered

    # ------------------------------------------------------------------
    # Linear interpolation with sigma_min noise
    # ------------------------------------------------------------------
    def interpolate(
        self,
        x0_paired: Tensor,
        x1_paired: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Linear interpolation with sigma_min Gaussian noise.

        Args:
            x0_paired: OT-reindexed noisy patches, shape (B, 3, N).
            x1_paired: OT-reindexed clean patches, shape (B, 3, N).
            t: Timestep, shape (B, 1, 1) for broadcast over (B, 3, N).

        Returns:
            x_t: Interpolated sample, shape (B, 3, N).
        """
        epsilon: Tensor = torch.randn_like(x0_paired)
        x_t: Tensor = (1.0 - t) * x0_paired + t * x1_paired + self.sigma_min * epsilon
        return x_t

    # ------------------------------------------------------------------
    # Backbone call helper
    # ------------------------------------------------------------------
    def _predict_velocity(
        self,
        x_t: Tensor,
        t_flat: Tensor,
        x_cond: Tensor,
        use_ema: bool = False,
    ) -> Tensor:
        """Call the backbone with properly scaled timestep.

        Args:
            x_t:     Interpolated state, (B, 3, N).
            t_flat:  CFM time in [0, 1], shape (B,).
            x_cond:  Conditioning input (noisy), (B, 3, N).
            use_ema: Whether to use the EMA model.

        Returns:
            Predicted velocity, (B, 3, N).
        """
        t_scaled: Tensor = t_flat * self.timestep_scale  # scale to ~0–1000
        if use_ema and self.ema is not None:
            return self.ema(x_t, t_scaled, x_cond=x_cond)
        else:
            return self.model(x_t, t_scaled, x_cond=x_cond)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def train_loss(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        Full training loss for one minibatch.

        Pipeline:
          1. Point-level OT coupling  ->  (x0, x1_reordered)
          2. Sample t ~ Uniform(0, 1), shape (B, 1, 1)
          3. Interpolate  ->  x_t
          4. Predict velocity: v_pred = backbone(x_t, t, x0)
          5. Return MSE(v_pred, x1_reordered - x0)

        The OT step reorders x1's points to match x0's ordering, giving
        straight-line interpolation paths. x0 is unchanged and serves as the
        conditioning input.

        Args:
            x0: Noisy patches, shape (B, 3, N).
            x1: Clean patches, shape (B, 3, N).

        Returns:
            Scalar MSE loss tensor.
        """
        import time as _time

        B: int = x0.shape[0]
        _do_time = self._timing_countdown > 0

        # Step 1: Point-level OT coupling — reorder x1 points to match x0
        x0_paired, x1_paired = self.ot_coupling(x0, x1)

        # Step 2: sample t ~ Uniform(0, 1)
        t: Tensor = torch.rand(B, 1, 1, device=x0.device, dtype=x0.dtype)

        # Step 3: interpolate
        x_t: Tensor = self.interpolate(x0_paired, x1_paired, t)

        # Step 4: predict velocity — condition on x0 (unchanged)
        if _do_time:
            torch.cuda.synchronize()
            _t_fwd0 = _time.perf_counter()
        t_flat: Tensor = t[:, 0, 0]  # (B,)
        v_pred: Tensor = self._predict_velocity(x_t, t_flat, x_cond=x0_paired)
        if _do_time:
            torch.cuda.synchronize()
            _fwd_ms = (_time.perf_counter() - _t_fwd0) * 1000.0

        # Step 5: MSE loss
        target: Tensor = x1_paired - x0_paired
        loss: Tensor = (v_pred - target).pow(2).mean()

        if _do_time:
            self._timing_countdown -= 1
            logger.warning(
                "[timing] ot_coupling={:.1f}ms  forward={:.1f}ms  "
                "(ot_update_freq={})",
                self._last_ot_ms, _fwd_ms, self.ot_update_freq,
            )

        return loss

    # ------------------------------------------------------------------
    # forward() — compatible with existing training loop call signature
    # ------------------------------------------------------------------
    def forward(
        self,
        x0: Tensor,
        x1: Optional[Tensor] = None,
        x_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward step — matches the P2PB call signature.

        In the existing training loop the call is:
            loss = model(x_gt, x1=x_start, x_cond=x_cond)
        where x_gt = clean, x_start = noisy, x_cond = None (PUNet).

        For OT-CFM: x0_noisy = x1 (x_start), x1_clean = x0 (x_gt).

        Args:
            x0: Clean patches (x_gt), shape (B, 3, N).
            x1: Noisy patches (x_start), shape (B, 3, N).
            x_cond: Unused for PUNet (kept for API compat).

        Returns:
            Scalar MSE loss tensor.
        """
        # Swap: CFM x0 = noisy, CFM x1 = clean
        return self.train_loss(x1, x0)

    # ------------------------------------------------------------------
    # Inference — Euler ODE integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        x_cond: Optional[Tensor] = None,
        x_start: Optional[Tensor] = None,
        clip: bool = False,
        use_ema: bool = False,
        verbose: bool = True,
        log_count: int = 10,
        steps: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Denoise x_start via Euler integration of the learned ODE.

        No OT coupling is performed at inference. x_start is used directly as
        the conditioning input to the backbone at every integration step.

        Args:
            x_cond:   Unused for PUNet (kept for API compat).
            x_start:  Noisy patches, shape (B, 3, N).
            clip:     Unused (kept for API compat).
            use_ema:  Whether to use the EMA model.
            verbose:  Unused (kept for API compat).
            log_count: Number of intermediate steps to log.
            steps:    Number of function evaluations (Euler steps).

        Returns:
            Dict with keys 'x_chain', 'x_pred', 'x_start'.
        """
        nfe: int = steps if steps is not None else 10
        B: int = x_start.shape[0]
        dt: float = 1.0 / nfe

        self.model.eval()

        x_t: Tensor = x_start.clone()
        xs: list[Tensor] = []

        for i in range(nfe):
            t_val: float = i / nfe
            t_batch: Tensor = torch.full(
                (B,), t_val, device=x_start.device, dtype=x_start.dtype
            )
            v: Tensor = self._predict_velocity(
                x_t, t_batch, x_cond=x_start, use_ema=use_ema
            )
            x_t = x_t + v * dt
            xs.append(x_t.clone())

        # Stack intermediate steps: (B, nfe, 3, N)
        xs_tensor: Tensor = torch.stack(xs, dim=1)
        # Flip so index 0 = final prediction (matches DDPM convention used
        # by evaluation.py and denoise_object.py).
        xs_tensor = torch.flip(xs_tensor, dims=(1,))

        self.model.train()

        return {
            "x_chain": xs_tensor,
            "x_pred": x_t.transpose(1, 2),  # [B, N, 3] — matches DDPM convention
            "x_start": x_start,
        }


class LatentOTFlowBridge(OTFlowBridge):
    """OT-CFM flow bridge with frozen AE + FreqEncodingTransformer latent conditioning.

    Pipeline:
      - Frozen GeomTokenAE encodes noisy points -> [B, M=64, 512] latent tokens
      - Trainable FreqEncodingTransformer refines them (timestep-conditioned)
      - Latent tokens injected into backbone via LatentWriteAttention (cross-attn)
      - OT coupling + CFM loss on backbone output (unchanged from base OTFlowBridge)

    The backbone receives latent tokens as x_cond instead of raw noisy points.
    """

    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        from models.autoencoder import GeomTokenAE
        from models.freq_encoding_transformer import FreqEncodingTransformer

        # ---- Load frozen AE ----
        self.ae = GeomTokenAE(
            in_dim=getattr(cfg.model, "in_dim", 3),
            feat_dim=getattr(cfg.model, "feat_dim", 512),
            use_dino=getattr(cfg.model, "use_dino", False),
            use_offset_attn=getattr(cfg.model, "use_offset_attn", True),
        )

        ae_ckpt_path = getattr(cfg.model, "ae_ckpt", "")
        try:
            ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            self.ae.load_state_dict(ckpt, strict=False)
            logger.warning(f"[LatentOTFlowBridge] AE checkpoint loaded: {ae_ckpt_path}")
        except Exception as e:
            logger.warning(f"Failed to load AE checkpoint from {ae_ckpt_path}: {e}")

        for p in self.ae.parameters():
            p.requires_grad = False
        self.ae.eval()

        ae_total = sum(p.numel() for p in self.ae.parameters())
        ae_trainable = sum(p.numel() for p in self.ae.parameters() if p.requires_grad)
        logger.warning(
            f"[LatentOTFlowBridge] AE FROZEN: {ae_total:,} params total, "
            f"{ae_trainable} trainable (expected 0). "
            f"requires_grad=False on ALL AE params. OK"
        )

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
        ft_total = sum(p.numel() for p in self.freq_transformer.parameters())
        logger.warning(
            f"[LatentOTFlowBridge] FreqEncodingTransformer TRAINABLE: {ft_total:,} params."
        )

        # ---- Summary ----
        backbone_total = sum(p.numel() for p in self.model.parameters())
        backbone_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        write_attn_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if "write_attn" in n and p.requires_grad
        )
        logger.warning(
            f"[LatentOTFlowBridge] Parameter summary: "
            f"Backbone {backbone_total:,} total / {backbone_trainable:,} trainable | "
            f"write_attn {write_attn_params:,} | "
            f"FreqTransformer {ft_total:,} (trainable) | "
            f"AE {ae_total:,} (FROZEN, 0 trainable)"
        )

    def _to_b3n(self, x):
        """Ensure [B,3,N] layout."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D, got {x.shape}")
        B, D1, D2 = x.shape
        if D1 == 3:
            return x
        if D2 == 3:
            return x.transpose(1, 2).contiguous()
        if D1 > 3:
            return x[:, :3, :].contiguous()
        raise RuntimeError(f"Cannot determine xyz layout from {x.shape}")

    @torch.no_grad()
    def _encode_ae(self, x_b3n):
        """Run frozen AE encoder. Returns [B, M, 512] latent tokens."""
        self.ae.eval()
        return self.ae.encode(x_b3n)

    def _get_latent_cond(self, x_noisy_b3n, t_flat):
        """Encode noisy points through AE + FreqTransformer.

        Args:
            x_noisy_b3n: [B, 3, N] noisy points
            t_flat:      [B] CFM time in [0, 1]
        Returns:
            cond_tokens: [B, M, D] latent conditioning tokens
        """
        ae_latent = self._encode_ae(x_noisy_b3n)  # [B, M, 512]
        # Scale t to ~0-1000 for the sinusoidal embedding in FreqTransformer
        t_scaled = t_flat * self.timestep_scale
        cond_tokens = self.freq_transformer(ae_latent, t=t_scaled)
        return cond_tokens, ae_latent

    # ------------------------------------------------------------------
    # Override _predict_velocity to use latent tokens instead of raw points
    # ------------------------------------------------------------------
    def _predict_velocity_latent(self, x_t, t_flat, cond_tokens, use_ema=False):
        """Call backbone with latent conditioning tokens.

        Args:
            x_t:          [B, 3, N] interpolated state
            t_flat:       [B] CFM time in [0, 1]
            cond_tokens:  [B, M, D] latent tokens from FreqTransformer
            use_ema:      whether to use EMA model

        Returns:
            Predicted velocity [B, 3, N].
        """
        t_scaled = t_flat * self.timestep_scale
        if use_ema and self.ema is not None:
            return self.ema(x_t, t_scaled, x_cond=cond_tokens)
        else:
            return self.model(x_t, t_scaled, x_cond=cond_tokens)

    # ------------------------------------------------------------------
    # Training loss — overrides the base class
    # ------------------------------------------------------------------
    def train_loss(self, x0, x1):
        """
        Full training loss: OT-CFM with latent conditioning.

        Args:
            x0: Noisy patches [B, 3+extra, N] or [B, 3, N].
                For ScanNet++ DINO, x0 contains xyz + DINO as extra channels.
            x1: Clean patches [B, 3, N] — xyz only.

        Returns:
            Scalar loss tensor.
        """
        import time as _time

        # Separate DINO (or any extra) channels from noisy xyz before OT + interpolation.
        # The backbone still receives the full x_t (xyz + extra) via x_t_backbone.
        extra = None
        if x0.shape[1] > 3:
            extra = x0[:, 3:, :].contiguous()   # e.g. DINO [B, 384, N]
            x0 = x0[:, :3, :].contiguous()       # xyz only [B, 3, N]

        B = x0.shape[0]
        _do_time = self._timing_countdown > 0

        # Step 1: Point-level OT coupling (xyz only)
        x0_paired, x1_paired = self.ot_coupling(x0, x1)

        # Step 2: sample t ~ Uniform(0, 1)
        t = torch.rand(B, 1, 1, device=x0.device, dtype=x0.dtype)
        t_flat = t[:, 0, 0]

        # Step 3: interpolate
        x_t = self.interpolate(x0_paired, x1_paired, t)

        # Step 4: encode noisy points -> latent conditioning tokens
        x0_b3n = self._to_b3n(x0_paired)
        cond_tokens, ae_latent_noisy = self._get_latent_cond(x0_b3n, t_flat)

        # Step 5: predict velocity — backbone receives full x_t (xyz + DINO if present)
        x_t_backbone = torch.cat([x_t, extra], dim=1) if extra is not None else x_t
        if _do_time:
            torch.cuda.synchronize()
            _t_fwd0 = _time.perf_counter()
        v_pred = self._predict_velocity_latent(x_t_backbone, t_flat, cond_tokens)
        if _do_time:
            torch.cuda.synchronize()
            _fwd_ms = (_time.perf_counter() - _t_fwd0) * 1000.0

        # Step 6: MSE loss — compare xyz velocity only
        target = x1_paired - x0_paired
        loss = (v_pred - target).pow(2).mean()

        # Auxiliary latent consistency loss: cosine sim between FreqTransformer
        # output and clean AE latent
        latent_loss_w = float(self.cfg.diffusion.get("latent_loss_weight", 0.0))
        if self.training and latent_loss_w > 0.0:
            x1_b3n = self._to_b3n(x1_paired)
            ae_latent_clean = self._encode_ae(x1_b3n)
            lat_pred = nn.functional.normalize(cond_tokens.mean(dim=1), dim=-1)
            lat_target = nn.functional.normalize(ae_latent_clean.mean(dim=1), dim=-1)
            latent_loss = (1.0 - (lat_pred * lat_target).sum(dim=-1)).mean()
            loss = loss + latent_loss_w * latent_loss

        if _do_time:
            self._timing_countdown -= 1
            logger.warning(
                "[timing] ot_coupling={:.1f}ms  forward={:.1f}ms",
                self._last_ot_ms, _fwd_ms,
            )

        return loss

    # ------------------------------------------------------------------
    # forward() — compatible with existing training loop
    # ------------------------------------------------------------------
    def forward(self, x0, x1=None, x_cond=None):
        """Forward step.

        The training loop calls: loss = model(x_gt, x1=x_start, x_cond=x_cond)
        where x_gt=clean, x_start=noisy. OT-CFM: x0=noisy, x1=clean.
        """
        return self.train_loss(x1, x0)

    # ------------------------------------------------------------------
    # Inference — Euler ODE with latent conditioning
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, x_cond=None, x_start=None, clip=False, use_ema=False,
               verbose=True, log_count=10, steps=None):
        """Denoise x_start via Euler integration with latent conditioning."""
        nfe = steps if steps is not None else 10
        B = x_start.shape[0]
        dt = 1.0 / nfe

        self.model.eval()
        self.freq_transformer.eval()

        # Accept DINO features passed separately via x_cond (API compat with
        # denoise_room.py which mirrors the DDPM call convention).
        # Concatenate into x_start so the channel-split below works uniformly.
        if x_cond is not None and x_start.shape[1] == 3:
            x_start = torch.cat([x_start, x_cond], dim=1)  # [B, 3+D, N]

        # Separate DINO extra channels from xyz — DINO is static throughout ODE.
        extra = None
        if x_start.shape[1] > 3:
            extra = x_start[:, 3:, :].contiguous()           # [B, 384, N]
            x_noisy_b3n = x_start[:, :3, :].contiguous()     # [B, 3, N]
        else:
            x_noisy_b3n = self._to_b3n(x_start)

        # Encode AE latent once from the noisy xyz (fixed during inference)
        ae_latent = self._encode_ae(x_noisy_b3n)
        x_t = x_noisy_b3n.clone()  # ODE state is xyz-only [B, 3, N]

        xs = []
        for i in range(nfe):
            t_val = i / nfe
            t_batch = torch.full((B,), t_val, device=x_start.device, dtype=x_start.dtype)
            # FreqTransformer runs at each step with the current t
            t_scaled = t_batch * self.timestep_scale
            cond_tokens = self.freq_transformer(ae_latent, t=t_scaled)
            # Backbone input: xyz + DINO (if present)
            x_t_backbone = torch.cat([x_t, extra], dim=1) if extra is not None else x_t
            v = self._predict_velocity_latent(x_t_backbone, t_batch, cond_tokens, use_ema=use_ema)
            x_t = x_t + v * dt  # update xyz-only ODE state
            xs.append(x_t.clone())

        xs_tensor = torch.stack(xs, dim=1)
        xs_tensor = torch.flip(xs_tensor, dims=(1,))

        self.model.train()
        self.freq_transformer.train()

        return {
            "x_chain": xs_tensor,
            "x_pred": x_t.transpose(1, 2),  # [B, N, 3] — matches DDPM convention
            "x_start": x_start,
        }
