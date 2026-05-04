##### Frequency Encoding Transformer - ported from new_work_Hash_ICIP


from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention
from models.Transformers.Input_Encodings import FrequencyEncoding
from ..pvcnn import (
    Pnet2Stage,
    PVCData,
    SharedMLP,
    Swish,
    create_fp_components,
    create_mlp_components,
)

# New imports for transformer layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import grid_sample



class Frequency_Encoding_Transformer(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        return_layers: bool = False,
    ):
        super().__init__()

        model_cfg = cfg.model
        pvd_cfg = model_cfg.PVD

        # Initialize class variables
        self.return_layers = return_layers
        self.input_dim = train_utils.default(model_cfg.in_dim, 3)

        if "extra_feature_channels" in pvd_cfg:
            self.extra_feature_channels = pvd_cfg.extra_feature_channels
        elif "extra_feature_channels" in model_cfg:
            self.extra_feature_channels = model_cfg.extra_feature_channels
        else:
            self.extra_feature_channels = 0

        self.embed_dim = train_utils.default(model_cfg.time_embed_dim, 64)

        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)
        num_heads = train_utils.default(pvd_cfg.get("transformer_heads", 8), 8)
        num_encoder_layers = train_utils.default(pvd_cfg.get("transformer_layers", 6), 6)
        transformer_ffn_dim = train_utils.default(pvd_cfg.get("transformer_ffn_dim", 256), 256)

        self.embedf = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.global_pnet = None
        self.cond_emb_dim = 0

        self.f_embed_dim = pvd_cfg.get("feat_embed_dim", self.extra_feature_channels)

        # Feature embedding
        if self.f_embed_dim != self.extra_feature_channels or self.extra_feature_channels == 0:
            in_dim = self.extra_feature_channels if self.extra_feature_channels > 0 else self.input_dim
            self.embed_feats = nn.Sequential(
                nn.Conv1d(in_dim, self.f_embed_dim, kernel_size=1, bias=True),
                nn.GroupNorm(8, self.f_embed_dim),
                Swish(),
                nn.Conv1d(self.f_embed_dim, self.f_embed_dim, kernel_size=1, bias=True),
            )
        else:
            self.embed_feats = None

        # Positional Encoding
        self.pos_enc = FrequencyEncoding(self.f_embed_dim)
        
        # Transformer Encoder Layers
        # Updated d_model to match the concatenated feature dimensions (feat_embed_dim + embed_dim)
        transformer_d_model = self.f_embed_dim + self.embed_dim
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Enable batch first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output projection
        out_mlp = cfg.model.PVD.get("out_mlp", 128)
        layers, *_ = create_mlp_components(
            in_channels=transformer_d_model,  # Updated to match transformer output
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
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)

        (B, C, N), device = x.shape, x.device
        assert (
            C == self.input_dim + self.extra_feature_channels
        ), f"input dim: {C}, expected: {self.input_dim + self.extra_feature_channels}"

        coords = x[:, : self.input_dim, :].contiguous()  # [B, 3, N]
        features = x[:, self.input_dim :, :].contiguous()  # [B, C_feat, N]

        # Embed features
        if self.embed_feats is not None:
            if self.extra_feature_channels == 0:
                features = self.embed_feats(coords)
            else:
                features = self.embed_feats(features)

        # Positional Encoding
        pos_enc = self.pos_enc(coords)  # [B, dim, N]

        # Combine features and positional encoding
        features = features + pos_enc  # [B, dim, N]

        # Time embedding
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            time_emb = self.embedf(self.get_timestep_embedding(t, device))  # [B, embed_dim]
            time_emb = time_emb.unsqueeze(-1).expand(-1, -1, N)  # [B, embed_dim, N]
            features = torch.cat([features, time_emb], dim=1)  # [B, dim + embed_dim, N]

        # Prepare features for transformer
        features = features.permute(0, 2, 1)  # [B, N, C] where C = dim + embed_dim = 128

        # Transformer Encoder
        features = self.transformer_encoder(features)  # [B, N, C]

        # Prepare data for classifier
        features = features.permute(0, 2, 1)  # [B, C, N]

        # Output projection
        data = PVCData(features=features, coords=coords)
        for l in self.classifier:
            if isinstance(l, SharedMLP):
                data = l(data)
            else:
                data.features = l(data.features)

        return data.features  # [B, out_dim, N]
