"""
Evaluation script for GeomTokenAE (AE v2)
Tests reconstruction quality over:
  - clean_10k / clean_50k (subsampled to 10k for fair comparison)
  - noisy_10k / noisy_50k at noise levels 0.01, 0.02, 0.03
Reports per-shape and mean Chamfer Distance (same formula as training).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F

from models.autoencoder import GeomTokenAE

# ─────────────────────────────────────────────────────────────
# Chamfer distance  (identical formula used in training)
# ─────────────────────────────────────────────────────────────
def chamfer_distance_per_shape(x, y):
    """x, y: [B,N,3]  → returns [B] CD values"""
    diff = torch.cdist(x, y, p=2)          # [B,N,M]
    mins1 = diff.min(dim=2)[0].mean(dim=1) # [B]
    mins2 = diff.min(dim=1)[0].mean(dim=1) # [B]
    return mins1 + mins2                   # [B]


# ─────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────
def load_xyz(path):
    """Load .xyz file → numpy (N,3)"""
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(p) for p in parts[:3]])
    return np.array(pts, dtype=np.float32)


def normalise(pts):
    """Bounding-box centre + unit-sphere scale (same as make_npy_full_poisson.py)"""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pts = pts - (lo + hi) / 2.0
    r = np.linalg.norm(pts, axis=1).max()
    if r > 1e-8:
        pts = pts / r
    return pts


def subsample(pts, n=10000, seed=0):
    """Random subsample to exactly n points"""
    rng = np.random.default_rng(seed)
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


# ─────────────────────────────────────────────────────────────
# Evaluate one subset
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_subset(model, files, device, max_pts=10000, label=""):
    model.eval()
    cds = []
    for fpath in sorted(files):
        pts = load_xyz(fpath)
        if len(pts) == 0:
            print(f"  [WARN] Empty file: {fpath}")
            continue
        pts = normalise(subsample(pts, max_pts))
        x = torch.from_numpy(pts).float().unsqueeze(0).to(device)   # [1,N,3]
        recon, _ = model(x)                                           # [1,N,3]
        cd = chamfer_distance_per_shape(x, recon).item()
        name = os.path.splitext(os.path.basename(fpath))[0]
        cds.append(cd)
        print(f"  {name:30s}  CD = {cd:.6f}")
    mean_cd = float(np.mean(cds)) if cds else float("nan")
    print(f"  {'─'*46}")
    print(f"  {label:30s}  Mean CD = {mean_cd:.6f}  ({len(cds)} shapes)\n")
    return mean_cd, cds


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True,  help="Path to AE checkpoint (.pth)")
    parser.add_argument("--test_dir", required=True,  help="Path to AE_test root directory")
    parser.add_argument("--device",   default="cuda", help="cuda / cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────
    model = GeomTokenAE(
        in_dim=3, feat_dim=512,
        use_cross_attn=True, use_offset_attn=True
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # strip DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}\n")

    root = args.test_dir
    results = {}

    # ── Clean 10k ─────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(root, "clean_10k", "*.xyz")))
    print(f"{'═'*52}")
    print(f"  clean_10k  ({len(files)} shapes, no subsampling)")
    print(f"{'═'*52}")
    mean, _ = eval_subset(model, files, device, max_pts=10000, label="clean_10k")
    results["clean_10k"] = mean

    # ── Clean 50k (subsampled to 10k for fair comparison) ─────
    files = sorted(glob.glob(os.path.join(root, "clean_50k", "*.xyz")))
    print(f"{'═'*52}")
    print(f"  clean_50k  ({len(files)} shapes, subsampled to 10k)")
    print(f"{'═'*52}")
    mean, _ = eval_subset(model, files, device, max_pts=10000, label="clean_50k→10k")
    results["clean_50k_sub10k"] = mean

    # ── Noisy 10k at three levels ──────────────────────────────
    for level in ["noisy_0.01", "noisy_0.02", "noisy_0.03"]:
        files = sorted(glob.glob(os.path.join(root, "noisy_10k", level, "*.xyz")))
        label = f"noisy_10k/{level}"
        print(f"{'═'*52}")
        print(f"  {label}  ({len(files)} shapes)")
        print(f"{'═'*52}")
        mean, _ = eval_subset(model, files, device, max_pts=10000, label=label)
        results[f"noisy_10k_{level}"] = mean

    # ── Noisy 50k at three levels (subsampled to 10k) ─────────
    for level in ["noisy_0.01", "noisy_0.02", "noisy_0.03"]:
        noisy_path = os.path.join(root, "noisy_50k", level)
        if not os.path.isdir(noisy_path):
            continue
        files = sorted(glob.glob(os.path.join(noisy_path, "*.xyz")))
        label = f"noisy_50k/{level}→10k"
        print(f"{'═'*52}")
        print(f"  {label}  ({len(files)} shapes)")
        print(f"{'═'*52}")
        mean, _ = eval_subset(model, files, device, max_pts=10000, label=label)
        results[f"noisy_50k_{level}"] = mean

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print("  SUMMARY")
    print(f"{'═'*52}")
    for k, v in results.items():
        print(f"  {k:38s}  {v:.6f}")
    print(f"{'═'*52}\n")


if __name__ == "__main__":
    main()
