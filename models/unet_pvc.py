from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention

from .pvcnn import (
    LinearAttention,
    Pnet2Stage,
    PVCData,
    SharedMLP,
    Swish,
    create_fp_components,
    create_mlp_components,
    create_pvc_layer_params,
    create_sa_components,
)


class LatentWriteAttention(nn.Module):
    """Cross-attention Write layer: N spatial points query M latent tokens.

    queries  : per-point features  [B, query_dim, N]  (BCN layout)
    keys/vals: latent tokens        [B, M, kv_dim]     (BNC layout)
    returns  : per-point residual   [B, query_dim, N]  (BCN layout)

    Supports query_dim != kv_dim (e.g., shallow FP decoder querying 512-dim latent).
    Xavier-init output projection -> small residual at step 0.
    """

    def __init__(self, query_dim: int, kv_dim: int = None, num_heads: int = 8):
        super().__init__()
        kv_dim = kv_dim if kv_dim is not None else query_dim
        while query_dim % num_heads != 0:
            num_heads = num_heads // 2
        assert num_heads >= 1
        self.num_heads = num_heads
        self.head_dim  = query_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(kv_dim)
        self.to_q   = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k   = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_v   = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

        nn.init.xavier_uniform_(self.to_out.weight, gain=0.5)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, features_bcn: torch.Tensor, latent_bnc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_bcn: [B, query_dim, N] backbone features (BCN)
            latent_bnc:   [B, M, kv_dim]    latent tokens (BNC)
        Returns:
            residual:     [B, query_dim, N] (BCN)
        """
        B, D, N = features_bcn.shape
        q_in = features_bcn.permute(0, 2, 1)  # [B, N, D]
        q_in = self.norm_q(q_in)
        kv_in = self.norm_k(latent_bnc)

        Q = self.to_q(q_in)     # [B, N, D]
        K = self.to_k(kv_in)    # [B, M, D]
        V = self.to_v(kv_in)    # [B, M, D]

        h, d = self.num_heads, self.head_dim
        Q = Q.view(B, N, h, d).transpose(1, 2)             # [B, h, N, d]
        K = K.view(B, -1, h, d).transpose(1, 2)            # [B, h, M, d]
        V = V.view(B, -1, h, d).transpose(1, 2)            # [B, h, M, d]

        attn = (Q @ K.transpose(-2, -1)) * self.scale      # [B, h, N, M]
        attn = attn.softmax(dim=-1)
        out = attn @ V                                       # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)          # [B, N, D]
        out = self.to_out(out)                               # [B, N, D]
        return out.permute(0, 2, 1)                          # [B, D, N]


# adapted from https://github.com/alexzhou907/PVD
class PVCNN2Unet(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        return_layers: bool = False,
    ):
        super().__init__()

        model_cfg = cfg.model
        pvd_cfg = model_cfg.PVD

        # initialize class variables
        self.return_layers = return_layers
        self.input_dim = train_utils.default(model_cfg.in_dim, 3)

        if "extra_feature_channels" in pvd_cfg:
            self.extra_feature_channels = pvd_cfg.extra_feature_channels
        elif "extra_feature_channels" in model_cfg:
            self.extra_feature_channels = model_cfg.extra_feature_channels

        self.embed_dim = train_utils.default(model_cfg.time_embed_dim, 64)

        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)
        attn_type = train_utils.default(pvd_cfg.attention_type, "linear")

        self.embedf = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # global embedding / latent FiLM conditioning
        latent_film = pvd_cfg.get("latent_film", False)
        if pvd_cfg.get("use_global_embedding", False):
            self.cond_emb_dim = pvd_cfg.global_embedding_dim
            c = self.cond_emb_dim
            self.global_pnet = Pnet2Stage(
                [self.input_dim, c // 8, c // 4],
                [c // 2, c],
            )
        elif latent_film and pvd_cfg.get("film_cond_dim", 0) > 0:
            self.cond_emb_dim = pvd_cfg.film_cond_dim
            self.global_pnet = None
            self.latent_pool_proj = nn.Sequential(
                nn.Linear(pvd_cfg.film_cond_dim, pvd_cfg.film_cond_dim),
                nn.SiLU(),
                nn.Linear(pvd_cfg.film_cond_dim, pvd_cfg.film_cond_dim),
            )
        else:
            self.global_pnet = None
            self.cond_emb_dim = 0

        self.f_embed_dim = pvd_cfg.get("feat_embed_dim", self.extra_feature_channels)

        self.film_cond_dim = pvd_cfg.get("film_cond_dim", 0)

        self.embed_feats = None
        if self.f_embed_dim != self.extra_feature_channels:
            in_dim = self.extra_feature_channels
            if in_dim == 0:
                in_dim = self.input_dim
            self.embed_feats = nn.Sequential(
                nn.Conv1d(in_dim, self.f_embed_dim, kernel_size=1, bias=True),
                nn.GroupNorm(8, self.f_embed_dim),
                Swish(),
                nn.Conv1d(self.f_embed_dim, self.f_embed_dim, kernel_size=1, bias=True),
            )

        # Cross-attention Write conditioning: per-point features attend to latent tokens.
        latent_num_heads = pvd_cfg.get("latent_num_heads", 8)
        if self.film_cond_dim > 0 and self.f_embed_dim > 0:
            self.write_attn = LatentWriteAttention(self.f_embed_dim, kv_dim=self.film_cond_dim, num_heads=latent_num_heads)
        else:
            self.write_attn = None
        self._feat_dropout_p = float(pvd_cfg.get("feat_dropout_before_attn", 0.0))

        sa_blocks, fp_blocks = create_pvc_layer_params(
            npoints=cfg.data.npoints,
            channels=cfg.model.PVD.channels,
            n_sa_blocks=cfg.model.PVD.n_sa_blocks,
            n_fp_blocks=cfg.model.PVD.n_fp_blocks,
            radius=cfg.model.PVD.radius,
            voxel_resolutions=cfg.model.PVD.voxel_resolutions,
            centers=pvd_cfg.centers if "centers" in pvd_cfg else None,
        )

        # prepare attention
        if attn_type.lower() == "linear":
            attention_fn = partial(LinearAttention, heads=cfg.model.PVD.attention_heads)
        elif attn_type.lower() == "flash":
            attention_fn = partial(Attention, norm=False, flash=True, heads=cfg.model.PVD.attention_heads)
        else:
            attention_fn = None

        # create set abstraction layers
        (
            sa_layers,
            sa_in_channels,
            channels_sa_features,
            *_,
        ) = create_sa_components(
            input_dim=self.input_dim,
            sa_blocks=sa_blocks,
            extra_feature_channels=self.f_embed_dim,
            with_se=pvd_cfg.get("use_se", True),
            embed_dim=self.embed_dim,  # time embedding dim
            attention_fn=attention_fn,
            attention_layers=cfg.model.PVD.attentions,
            dropout=dropout,
            gn_groups=8,
            cond_dim=self.cond_emb_dim,
        )

        self.sa_layers = nn.ModuleList(sa_layers)

        if attention_fn is not None:
            self.global_att = attention_fn(dim=channels_sa_features)

        # create feature propagation layers
        # only use extra features in the last fp module WHY ACTUALLY??
        sa_in_channels[0] = self.f_embed_dim + self.input_dim

        fp_layers, channels_fp_features = create_fp_components(
            fp_blocks=fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=pvd_cfg.get("use_se", True),
            embed_dim=self.embed_dim,
            attention_layers=cfg.model.PVD.attentions,
            attention_fn=attention_fn,
            dropout=dropout,
            gn_groups=8,
            cond_dim=self.cond_emb_dim,
        )

        self.fp_layers = nn.ModuleList(fp_layers)

        # Per-FP-level write_attn: injects latent conditioning into the decoder at each resolution.
        fp_enable = pvd_cfg.get("fp_write_attn", True)
        if self.film_cond_dim > 0 and fp_enable:
            fp_channels_list = cfg.model.PVD.channels
            fp_out_dims = [
                fp_channels_list[3],  # FP0
                fp_channels_list[3],  # FP1
                fp_channels_list[2],  # FP2
                fp_channels_list[1],  # FP3
            ]
            self.fp_write_attns = nn.ModuleList([
                LatentWriteAttention(dim, kv_dim=self.film_cond_dim, num_heads=latent_num_heads)
                for dim in fp_out_dims
            ])
        else:
            self.fp_write_attns = None

        # output projection
        out_mlp = cfg.model.PVD.get("out_mlp", 128)
        layers, *_ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[out_mlp, dropout, out_dim],
            classifier=True,
            dim=2,
        )
        self.classifier = nn.ModuleList(layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        assert len(timesteps.shape) == 1, f"get shape: {timesteps.shape}"

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, x, t, x_cond=None):
        # x_cond is either:
        #  - [B, 3, N] noisy points for concat (legacy, extra_feature_channels=3)
        #  - [B, M, D] latent tokens for LatentWriteAttention (latent mode)
        # Detect latent mode: x_cond is [B, M, D] with M<<N and D>>3.
        latent_tokens = None
        if x_cond is not None:
            if x_cond.ndim == 3 and x_cond.shape[1] != x.shape[2] and x_cond.shape[2] > 3:
                # Latent tokens [B, M, D] — do NOT concat
                latent_tokens = x_cond
            else:
                # Legacy: concat noisy points
                x = torch.cat([x, x_cond], dim=1)

        (B, C, N), device = x.shape, x.device

        if not hasattr(self, "extra_feature_channels"):
            self.extra_feature_channels = 0
        assert (
            C == self.input_dim + self.extra_feature_channels
        ), f"input dim: {C}, expected: {self.input_dim + self.extra_feature_channels}"

        coords = x[:, : self.input_dim, :].contiguous()
        features = x[:, self.input_dim :, :].contiguous()

        # embed features if we set a feature embedding dimension
        if self.embed_feats is not None:
            if self.extra_feature_channels == 0:
                features = self.embed_feats(coords)
            else:
                features = self.embed_feats(features)

        # Cross-attention Write: per-point conditioning from latent tokens
        if latent_tokens is not None and self.write_attn is not None:
            feat_drop_p = getattr(self, '_feat_dropout_p', 0.0)
            if self.training and feat_drop_p > 0.0:
                features = F.dropout(features, p=feat_drop_p, training=True)
            features = features + self.write_attn(features, latent_tokens)

        # initialize data class
        data = PVCData(coords=coords, features=coords)

        # global embedding
        if self.global_pnet is not None:
            global_feature = self.global_pnet(data)
            data.cond = global_feature
        elif hasattr(self, 'latent_pool_proj') and latent_tokens is not None:
            cond_global = latent_tokens.mean(dim=1)
            data.cond = self.latent_pool_proj(cond_global)
        else:
            global_feature = None

        # take coords + extra features as the feature input to the model
        features = torch.cat([coords, features], dim=1)

        # initialize lists
        coords_list, in_features_list = [], []
        out_features_list = []

        # append concatenated coords and features to lists
        in_features_list.append(features)

        time_emb = None
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            time_emb = self.embedf(self.get_timestep_embedding(t, device))[:, :, None].expand(-1, -1, N)

        # initialize dataclass
        data.features = features
        data.time_emb = time_emb

        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(data.features)
            coords_list.append(data.coords)

            if i > 0 and data.time_emb is not None:
                data.features = torch.cat([data.features, data.time_emb], dim=1)
                data = sa_blocks(data)
            else:
                data = sa_blocks(data)

        # remove first added feature in feature list
        in_features_list.pop(1)

        # global attention at middle layer
        if self.global_att is not None:
            features = data.features
            if isinstance(self.global_att, LinearAttention):
                features = self.global_att(features)
            elif isinstance(self.global_att, Attention):
                features = rearrange(features, "b n c -> b c n")
                features = self.global_att(features)
                features = rearrange(features, "b c n -> b n c")
            else:
                raise ValueError(f"Invalid attention type: {type(self.global_att)}")
            data.features = features

        # add first element to out_features_list, after attention mechanism
        out_features_list.append(data.features)

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            data_fp = PVCData(
                features=in_features_list[-1 - fp_idx],
                coords=coords_list[-1 - fp_idx],
                lower_coords=data.coords,
                lower_features=(
                    torch.cat([data.features, data.time_emb], dim=1) if data.time_emb is not None else data.features
                ),
                time_emb=data.time_emb,
                cond=data.cond,
            )
            data = fp_blocks(data_fp)
            # Multi-level FP decoder conditioning: inject latent tokens at each resolution.
            if latent_tokens is not None and self.fp_write_attns is not None:
                data.features = data.features + self.fp_write_attns[fp_idx](data.features, latent_tokens)
            out_features_list.append(data.features)

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                data.features = l(data).features
            else:
                data.features = l(data.features)

        return data.features
