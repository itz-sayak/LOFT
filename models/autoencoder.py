import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch3d.ops
except ImportError:
    pytorch3d = None


# ---------------------------------------------------------------------
#  Geometry-aware feature propagation (PointNet++ FP style)
#  Interpolates sparse features → dense points using 3-NN inv-dist weighting.
#  Falls back to F.interpolate (index-linear) when pytorch3d unavailable.
# ---------------------------------------------------------------------
def fp_interpolate(xyz_dense, xyz_sparse, feats_sparse):
    """
    xyz_dense:    [B, 3, N]   — target (dense) point positions
    xyz_sparse:   [B, 3, M]   — source (sparse) point positions
    feats_sparse: [B, C, M]   — features at sparse points
    Returns:      [B, C, N]   — features interpolated at dense positions
    Uses 3-NN inverse-distance weighting.  Falls back to F.interpolate if
    pytorch3d is not available.
    """
    B, C, N = xyz_dense.shape[0], feats_sparse.shape[1], xyz_dense.shape[2]
    M = xyz_sparse.shape[2]
    K = min(3, M)

    if pytorch3d is not None:
        dists, idx, _ = pytorch3d.ops.knn_points(
            xyz_dense.permute(0, 2, 1).contiguous(),   # [B,N,3]
            xyz_sparse.permute(0, 2, 1).contiguous(),  # [B,M,3]
            K=K, return_nn=False,
        )
        # dists, idx: [B, N, K]
        dists = dists.clamp(min=1e-10)
        w = 1.0 / dists                                # [B, N, K]
        w = w / w.sum(dim=-1, keepdim=True)            # normalised weights

        # Memory-efficient gather: avoid O(B*N*M*C) expansion.
        # feats_sparse_bnc: [B, M, C]
        fp_bnc = feats_sparse.permute(0, 2, 1).contiguous()  # [B, M, C]
        idx_flat = idx.reshape(B, -1)                         # [B, N*K]
        # index along M dim — idx_flat is [B, N*K], expand to [B, N*K, C]
        idx_gather = idx_flat.unsqueeze(-1).expand(B, N * K, C)
        gathered_flat = torch.gather(fp_bnc, 1, idx_gather)  # [B, N*K, C]
        gathered = gathered_flat.view(B, N, K, C)             # [B, N, K, C]

        interpolated = (gathered * w.unsqueeze(-1)).sum(dim=2)  # [B, N, C]
        return interpolated.permute(0, 2, 1).contiguous()       # [B, C, N]

    # --- fallback: linear interpolation in index space (old behaviour) ---
    return F.interpolate(feats_sparse, size=N, mode="nearest")


# ---------------------------------------------------------------------
#  Shared MLP (PointNet-style 1×1 convolutions)
# ---------------------------------------------------------------------
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        layers = []
        last = in_channels
        for oc in out_channels_list:
            layers += [nn.Conv1d(last, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(inplace=True)]
            last = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [B,C,N]
        assert x.ndim == 3, f"SharedMLP: expected 3D input, got {x.shape}"
        return self.net(x)


# ---------------------------------------------------------------------
#  Set-Abstraction (simple random sampling + shared MLP)
# ---------------------------------------------------------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, in_channels, mlp):
        super().__init__()
        self.npoint = npoint
        self.mlp = SharedMLP(in_channels, mlp)

    def _sample_indices(self, coords):
        B, _, N = coords.shape
        npoint = min(self.npoint, N)

        if pytorch3d is not None and npoint < N:
            sampled, idx = pytorch3d.ops.sample_farthest_points(coords.transpose(1, 2).contiguous(), K=npoint)
            return idx, sampled.transpose(1, 2).contiguous()

        base_idx = torch.linspace(0, N - 1, steps=npoint, device=coords.device)
        idx = base_idx.round().long().unsqueeze(0).expand(B, -1)
        gathered = torch.gather(coords, 2, idx.unsqueeze(1).expand(-1, coords.shape[1], -1))
        return idx, gathered

    def forward(self, coords, features=None):  # coords [B,3,N], features [B,C,N]
        if features is None:
            features = coords

        idx, new_coords = self._sample_indices(coords)
        new_feat_in = torch.gather(features, 2, idx.unsqueeze(1).expand(-1, features.shape[1], -1))
        new_feat = self.mlp(new_feat_in)
        return new_coords, new_feat


# ---------------------------------------------------------------------
#  Feature-Propagation (upsampling MLP)
# ---------------------------------------------------------------------
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp):
        super().__init__()
        self.mlp = SharedMLP(in_channels, mlp)

    def forward(self, x):  # [B,C,N]
        assert x.ndim == 3, f"FeatureProp: expected 3D input, got {x.shape}"
        return self.mlp(x)


# ---------------------------------------------------------------------
#  Offset-Attention (PCT-style)
# ---------------------------------------------------------------------
class OffsetAttention(nn.Module):
    def __init__(self, dim, heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.ffn  = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.bn   = nn.BatchNorm1d(dim)
        self.do   = nn.Dropout(proj_drop)

    def forward(self, x):  # [B,N,C]
        y, _ = self.attn(x, x, x)
        off = x - y
        B, N, C = off.shape
        z = self.ffn(off).view(B * N, C)
        z = self.bn(z).view(B, N, C)
        z = self.do(z)
        return x + z


# ---------------------------------------------------------------------
#  Cross-Attention with gating
# ---------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.do   = nn.Dropout(proj_drop)

    def forward(self, q, kv):  # both [B,N,C]
        y, _ = self.attn(q, kv, kv)
        y = self.do(self.proj(y))
        return q + self.gate * y


# ---------------------------------------------------------------------
#  Squeeze-Excitation
# ---------------------------------------------------------------------
class SE(nn.Module):
    def __init__(self, C, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(C, C // r), nn.ReLU(inplace=True),
            nn.Linear(C // r, C), nn.Sigmoid()
        )

    def forward(self, x):  # [B,C,N]
        s = x.mean(dim=2)
        w = self.fc(s).unsqueeze(-1)
        return x * w


# ---------------------------------------------------------------------
#  Semantic-Attention Latent Autoencoder (AE-3, diffusion-ready)
# ---------------------------------------------------------------------
class SemanticAutoencoder(nn.Module):
    def __init__(
        self,
        in_dim=3,
        feat_dim=512,
        use_dino=False,
        dino_dim=384,
        sem_dim=128,
        use_cross_attn=True,
        use_offset_attn=True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_offset_attn = use_offset_attn
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.cond_enabled = True

        # Dynamically set first-layer input channels (handles conditioning or latent concat)
        self.extra_in_dim = feat_dim  # latent cond from diffusion (~512)
        total_in = in_dim + self.extra_in_dim

        # ---------------- Encoder ----------------
        self.sa1 = PointNetSetAbstraction(1024, total_in, [64, 128])
        self.sa2 = PointNetSetAbstraction(256, 128, [128, 256])
        self.sa3 = PointNetSetAbstraction(64, 256, [256, feat_dim])

        self.se1, self.se2, self.se3 = SE(128), SE(256), SE(feat_dim)
        self.mid_tr = OffsetAttention(256, heads=4)

        if self.use_offset_attn:
            self.bn_tr = nn.Sequential(
                OffsetAttention(feat_dim, heads=4),
                OffsetAttention(feat_dim, heads=4)
            )

        if self.use_cross_attn:
            self.cross1 = CrossAttention(128, heads=4)
            self.cross2 = CrossAttention(256, heads=4)

        # ---------------- Skip projections ----------------
        self.proj_l2 = nn.Conv1d(256, feat_dim, 1)
        self.proj_l1 = nn.Conv1d(128, feat_dim // 2, 1)

        # ---------------- Decoder ----------------
        self.fp3 = PointNetFeaturePropagation(feat_dim, [feat_dim, 256])
        self.fp2 = PointNetFeaturePropagation(256 + feat_dim // 2, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 64])

        self.refine = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1)
        )
        self.final_head = nn.Conv1d(64, 3, 1)

        # ---------------- Time embedding (FiLM) ----------------
        self.time_mlp = nn.Sequential(nn.Linear(1, feat_dim), nn.SiLU(), nn.Linear(feat_dim, feat_dim))
        self.time_gamma = nn.Linear(feat_dim, feat_dim)
        self.time_beta  = nn.Linear(feat_dim, feat_dim)

    # ------------------------------------------------------------
    def _ensure_bcn(self, xyz):
        assert xyz.ndim == 3
        return xyz.transpose(1, 2).contiguous() if xyz.shape[1] != 3 else xyz.contiguous()

    # ------------------------------------------------------------
    def forward(self, *args, **kwargs):
        xyz = args[0] if len(args) > 0 else kwargs.get("xyz")
        t = kwargs.get("t", None)
        x_cond = kwargs.get("x_cond", None)

        x = self._ensure_bcn(xyz)   # [B,3,N]
        B, _, N = x.shape

        # if conditioning present, concatenate to input
        if x_cond is not None:
            if x_cond.shape[1] == N:
                x_cond_bcn = x_cond.transpose(1, 2).contiguous()
            else:
                x_cond_bcn = x_cond
            xin = torch.cat([x, x_cond_bcn], dim=1)  # [B,3+Cs,N]
        else:
            # add dummy zeros if not provided (for consistent dims)
            xin = torch.cat([x, torch.zeros(B, self.extra_in_dim, N, device=x.device)], dim=1)

        # ---------------- Encoder ----------------
        l1_xyz, l1 = self.sa1(x, xin)
        l1 = self.se1(l1)

        if self.use_cross_attn and x_cond is not None:
            l1_tokens = l1.transpose(1, 2)
            c1_tokens = F.interpolate(x_cond_bcn, size=l1.shape[-1], mode="nearest").transpose(1, 2)
            if c1_tokens.shape[-1] != l1_tokens.shape[-1]:
                if not hasattr(self, "_proj1"):
                    self._proj1 = nn.Linear(c1_tokens.shape[-1], l1_tokens.shape[-1]).to(l1_tokens.device)
                c1_tokens = self._proj1(c1_tokens)
            l1_tokens = self.cross1(l1_tokens, c1_tokens)
            l1 = l1_tokens.transpose(1, 2).contiguous()

        l2_xyz, l2 = self.sa2(l1_xyz, l1)
        l2 = self.se2(l2)

        l2_tokens = l2.transpose(1, 2).contiguous()
        l2_tokens = self.mid_tr(l2_tokens)
        l2 = l2_tokens.transpose(1, 2).contiguous()

        if self.use_cross_attn and x_cond is not None:
            c2_tokens = F.interpolate(x_cond_bcn, size=l2.shape[-1], mode="nearest").transpose(1, 2)
            if c2_tokens.shape[-1] != l2_tokens.shape[-1]:
                if not hasattr(self, "_proj2"):
                    self._proj2 = nn.Linear(c2_tokens.shape[-1], l2_tokens.shape[-1]).to(l2_tokens.device)
                c2_tokens = self._proj2(c2_tokens)
            l2_tokens = self.cross2(l2_tokens, c2_tokens)
            l2 = l2_tokens.transpose(1, 2).contiguous()

        l3_xyz, l3 = self.sa3(l2_xyz, l2)
        l3 = self.se3(l3)
        latent = l3.transpose(1, 2).contiguous()

        # Time embedding (FiLM)
        if t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32, device=latent.device)
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            t_embed = self.time_mlp(t.float())
            gamma = self.time_gamma(t_embed)[:, None, :]
            beta  = self.time_beta(t_embed)[:, None, :]
            latent = (1 + gamma) * latent + beta

        # Bottleneck attention
        if self.use_offset_attn:
            latent = self.bn_tr(latent)

        # ---------------- Decoder (geometry-aware FP) ----------------
        # latent: [B,64,512], l3_xyz: [B,3,64]
        # l2:     [B,256,256], l2_xyz: [B,3,256]
        # l1:     [B,128,1024], l1_xyz: [B,3,1024]
        # x (original coords): [B,3,N]
        lat_feat = latent.transpose(1, 2)                          # [B,512,64]
        x_up = fp_interpolate(l2_xyz, l3_xyz, lat_feat)           # [B,512,256]
        x_up = self.fp3(x_up + self.proj_l2(l2))                  # [B,256,256]

        x_up = fp_interpolate(l1_xyz, l2_xyz, x_up)               # [B,256,1024]
        l1_proj = self.proj_l1(l1)                                 # [B,256,1024]
        x_up = torch.cat([x_up, l1_proj], dim=1)                  # [B,512,1024]
        x_up = self.fp2(x_up)                                      # [B,128,1024]

        x_up = fp_interpolate(x, l1_xyz, x_up)                    # [B,128,N]
        x_up = self.fp1(x_up)                                      # [B,64,N]
        x_up = x_up + self.refine(x_up)
        recon_feat = self.final_head(x_up)
        recon = recon_feat.transpose(1, 2).contiguous()

        assert recon.shape[-1] == 3, f"Output shape invalid: {recon.shape}"
        return recon, latent

    # ------------------------------------------------------------
    @torch.no_grad()
    def encode(self, xyz, **kwargs):
        _, z = self.forward(xyz, **kwargs)
        return z

    @torch.no_grad()
    def decode(self, latent, N=2048):
        assert latent.ndim == 3, f"Decode expects [B,M,C], got {latent.shape}"
        x_up = latent.transpose(1, 2)
        x_up = F.interpolate(x_up, size=256, mode="nearest")
        x_up = PointNetFeaturePropagation(self.feat_dim, [self.feat_dim, 256])(x_up)
        x_up = F.interpolate(x_up, size=1024, mode="nearest")
        x_up = PointNetFeaturePropagation(256, [256, 128])(x_up)
        x_up = F.interpolate(x_up, size=N, mode="nearest")
        x_up = PointNetFeaturePropagation(128, [128, 128, 64])(x_up)
        x_up = x_up + self.refine(x_up)
        out = self.final_head(x_up).transpose(1, 2).contiguous()
        return out

