"""
Frequency Encoder Transformer for conditioning the P2P-Bridge diffusion model
with AE latent vectors.

Pipeline:
    AE latent [B, M, C], timestep t [B]
        -> FourierFreqEncoding    (project to num_freqs bottleneck, compute sin/cos,
                                   concat with original, project back to C)
        -> + learnable positional embedding [1, M, C]
        -> N × (TransformerEncoderLayer + TimestepFiLM)   (pre-norm self-attention + FFN
                                                           with per-layer adaptive scale/shift)
        -> LayerNorm / linear output projection
        -> [B, M, out_dim]         (defaults: M=64, C=out_dim=512)

When t=None the module falls back to the original t-agnostic behaviour (FiLM is identity),
so existing checkpoints can still be loaded and run without modification.

The AE encoder is always frozen; only FreqEncodingTransformer is trainable.
"""

import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Sinusoidal timestep embedding (matches PVCNN2Unet convention)
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard DDPM sinusoidal position embedding.

    Args:
        timesteps: [B] integer or float timesteps / noise-levels
        dim:       embedding dimensionality (must be even)
    Returns:
        [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
#  TimestepFiLM — adaptive scale + shift conditioned on t
# ---------------------------------------------------------------------------

class TimestepFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on diffusion timestep.

    Given a sinusoidal embedding of t it predicts per-channel (γ, β) and applies:
        x' = γ · x + β

    Initialised so that γ≈1, β≈0 (identity at init) to not break pre-trained
    weights when fine-tuning.
    """

    def __init__(self, latent_dim: int, time_embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        # Zero-init final layer so γ≈1, β≈0 at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, M, C]  latent tokens
            t_emb: [B, D]     timestep embedding
        Returns:
            modulated [B, M, C]
        """
        gamma_beta = self.mlp(t_emb)               # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)   # each [B, C]
        # shift gamma so identity at init: gamma_eff = 1 + gamma
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
#  Fourier Frequency Encoding
# ---------------------------------------------------------------------------

class FourierFreqEncoding(nn.Module):
    """
    Spectral feature encoder applied per latent token.

    For each input token z ∈ R^C it:
      1. Projects z to a compact frequency space: f = z W_freq^T  ∈ R^{num_freqs}
      2. Computes [sin(f), cos(f)] ∈ R^{2·num_freqs}
      3. Concatenates [z, sin(f), cos(f)] ∈ R^{C + 2·num_freqs}
      4. Projects back to R^C with a learnable linear + LayerNorm

    The projection weights W_freq are learnable by default (set learnable=False
    for random fixed Fourier features à la RFF/NeRF).

    Args:
        latent_dim:  dimensionality C of each input token
        num_freqs:   number of frequency bands (bottleneck width)
        learnable:   whether W_freq is learnable (True) or fixed random (False)
    """

    def __init__(self, latent_dim: int, num_freqs: int = 6, learnable: bool = True):
        super().__init__()
        self.num_freqs = num_freqs

        # Frequency projection: R^C -> R^{num_freqs}
        W = torch.randn(num_freqs, latent_dim) * (1.0 / math.sqrt(latent_dim))
        if learnable:
            self.W_freq = nn.Parameter(W)
        else:
            self.register_buffer("W_freq", W)

        # Re-projection: R^{C + 2*num_freqs} -> R^C
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + 2 * num_freqs, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: latent tokens  [B, M, C]
        Returns:
            frequency-enriched tokens [B, M, C]
        """
        # Project to frequency bottleneck: [B, M, num_freqs]
        freq = x @ self.W_freq.T
        # Compute sinusoidal features: each [B, M, num_freqs]
        sin_f = torch.sin(freq)
        cos_f = torch.cos(freq)
        # Concatenate and project back: [B, M, C+2*num_freqs] -> [B, M, C]
        x_aug = torch.cat([x, sin_f, cos_f], dim=-1)
        return self.proj(x_aug)


# ---------------------------------------------------------------------------
#  Frequency Encoding Transformer (timestep-conditioned)
# ---------------------------------------------------------------------------

class FreqEncodingTransformer(nn.Module):
    """
    Refines AE latent tokens with frequency-aware, timestep-conditioned
    self-attention.

    Architecture:
        1. FourierFreqEncoding     – injects multi-frequency spectral features
        2. Learnable pos. embedding – distinguishes token positions
        3. N × (TransformerEncoderLayer + TimestepFiLM)
           – pre-norm self-attention + FFN, then adaptive scale/shift from t
        4. Output LayerNorm (+ linear if out_dim ≠ latent_dim)

    When ``t`` is not supplied to :meth:`forward`, the FiLM layers act as
    identity (γ=1, β=0), preserving backward-compatibility with checkpoints
    trained without timestep conditioning.

    Args:
        latent_dim:     input / internal hidden dimension (AE feat_dim, default 512)
        num_tokens:     expected number of latent tokens from AE (default 64)
        num_freqs:      Fourier frequency bands in FourierFreqEncoding (default 6)
        num_layers:     depth of the TransformerEncoder (default 4)
        num_heads:      multi-head attention heads (default 8)
        ff_mult:        FFN expansion ratio (default 4 → FFN dim = 4 × latent_dim)
        dropout:        attention + FFN dropout probability (default 0.1)
        out_dim:        output token dimension; None defaults to latent_dim
        learnable_freq: whether frequency projection weights are learned (True)
        time_embed_dim: sinusoidal timestep embedding size (default 64, matching PVCNN2Unet)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_tokens: int = 64,
        num_freqs: int = 6,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        out_dim: Optional[int] = None,
        learnable_freq: bool = True,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        out_dim = out_dim or latent_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.time_embed_dim = time_embed_dim

        # ---- 1. Fourier frequency encoding ----
        self.freq_enc = FourierFreqEncoding(
            latent_dim=latent_dim,
            num_freqs=num_freqs,
            learnable=learnable_freq,
        )

        # ---- 2. Learnable positional embedding ----
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- 3. Transformer layers with per-layer TimestepFiLM ----
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=latent_dim * ff_mult,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.film_layers.append(TimestepFiLM(latent_dim, time_embed_dim))

        # ---- 4. Timestep embedding MLP (sinusoidal → learned) ----
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ---- 5. Output head ----
        if out_dim != latent_dim:
            self.out_proj: nn.Module = nn.Sequential(
                nn.Linear(latent_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
        else:
            self.out_proj = nn.LayerNorm(latent_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z: AE latent tokens  [B, M, latent_dim]
               (output of SemanticAutoencoder.encode(), shape [B,64,512] by default)
            t: diffusion timestep  [B] (integer indices or continuous noise-levels).
               When None the FiLM layers act as identity (backward-compatible).
        Returns:
            refined tokens  [B, M, out_dim]
        """
        # ---- timestep embedding ----
        if t is not None:
            t_emb = _sinusoidal_embedding(t, self.time_embed_dim)  # [B, D]
            t_emb = self.time_mlp(t_emb)                           # [B, D]
        else:
            t_emb = None

        # ---- frequency encoding ----
        x = self.freq_enc(z)                    # [B, M, latent_dim]

        # ---- positional embedding (interpolate if M differs from init) ----
        pos = self.pos_embed
        if x.shape[1] != pos.shape[1]:
            pos = F.interpolate(
                pos.transpose(1, 2), size=x.shape[1], mode="nearest"
            ).transpose(1, 2)
        x = x + pos                             # [B, M, latent_dim]

        # ---- transformer layers + per-layer FiLM ----
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x)                        # [B, M, latent_dim]
            if t_emb is not None:
                x = film(x, t_emb)              # adaptive scale/shift

        # ---- output projection ----
        return self.out_proj(x)                 # [B, M, out_dim]
