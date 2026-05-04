### Transformer with decoder
import torch
import torch.nn as nn
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention

from ..pvcnn import (
    Pnet2Stage,
    PVCData,
    SharedMLP,
    Swish,
    create_fp_components,
    create_mlp_components,
)

class FrequencyEncoding(nn.Module):
    def __init__(self, dim: int):
        """
        Standard sinusoidal positional encoding with scaled L2 norm.
        Args:
            dim: Desired output dimension (should be 64 for this implementation).
        """
        super().__init__()
        print("Using standard sinusoidal positional encoding with scaled L2 norm.")

        self.dim = dim  # Desired output dimension (e.g., 64)
        print(f"Output dimension (dim): {self.dim}")

        # Compute the number of frequency bands L
        # The total dimension D is given by D = 4 + 6L
        # Solve for L: L = (dim - 4) // 6
        assert (self.dim - 4) % 6 == 0, "dim - 4 must be divisible by 6"
        self.L = (self.dim - 4) // 6
        print(f"Number of frequency bands (L): {self.L}")

        # Frequencies: [2^0, 2^1, ..., 2^{L-1}]
        self.freq_bands = 2.0 ** torch.linspace(0, self.L - 1, self.L)
        print(f"Frequency bands: {self.freq_bands.tolist()}")

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, 3, N], representing coordinates.

        Returns:
            embeddings: Tensor of shape [B, dim, N], positional encodings.
        """
        B, C, N = coords.shape
        assert C == 3, "Input coordinates should have 3 channels (x, y, z)"
        device = coords.device

        # Normalize coordinates to [0, 1]
        coords_min = coords.amin(dim=2, keepdim=True)
        coords_max = coords.amax(dim=2, keepdim=True)
        coords_range = coords_max - coords_min + 1e-6  # Avoid division by zero
        coords_norm = (coords - coords_min) / coords_range  # [B, 3, N]

        # Compute scaled L2 norm of coordinates
        l2_norm = torch.norm(coords_norm, dim=1, keepdim=True)  # [B, 1, N]
        # Scale L2 norm to [0, 1]
        l2_norm_min = l2_norm.amin(dim=2, keepdim=True)
        l2_norm_max = l2_norm.amax(dim=2, keepdim=True)
        l2_norm_range = l2_norm_max - l2_norm_min + 1e-6
        l2_norm_scaled = (l2_norm - l2_norm_min) / l2_norm_range  # [B, 1, N]

        # Prepare frequency bands
        freq_bands = self.freq_bands.to(device)  # [L]

        # Compute sinusoidal encodings for each coordinate
        embeddings = [coords_norm]  # Start with normalized coordinates [B, 3, N]

        for freq in freq_bands:
            for fn in [torch.sin, torch.cos]:
                embeddings.append(fn(coords_norm * freq * np.pi))  # [B, 3, N]

        # Concatenate all embeddings
        embeddings = torch.cat(embeddings, dim=1)  # [B, 3 + 2*L*3, N]

        # Append scaled L2 norm
        embeddings = torch.cat([embeddings, l2_norm_scaled], dim=1)  # [B, D, N], D = 4 + 6L

        # Ensure the output dimension matches self.dim
        assert embeddings.shape[1] == self.dim, f"Output dimension {embeddings.shape[1]} does not match expected {self.dim}"

        return embeddings  # [B, dim, N]
