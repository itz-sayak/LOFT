"""
Generate a denoising-over-time GIF for LOFT, matching the style of
P2P-Bridge-OT-real-latent/assets/room-denoise.gif.

Usage (from /mnt/zone/B/NEW):
    conda run -n deepfill \
        python3 -u make_loft_gif.py \
        --scene 7bc286c1b6 \
        --steps 20 \
        --out /mnt/zone/B/NEW/P2P-Bridge-OT-real-latent/assets/loft-denoise.gif

Frame 0  = noisy input
Frame 1..steps  = progressive denoising (noisy → clean)
Color    = distance to GT (teal-green=close, magenta=far)
View     = orthographic top-down
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.spatial import cKDTree

# ── Make project importable ──────────────────────────────────────────────────
PROJ = "/mnt/zone/B/NEW/P2P-Bridge-OT-real-latent"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import omegaconf
import fpsample
from einops import rearrange
from sklearn import neighbors
from loguru import logger
from tqdm import tqdm
from third_party.pvcnn.functional.sampling import furthest_point_sample
from models.model_loader import load_diffusion
from denoise_room import (
    denoise_patch_batch,
    create_patches,
    update_prediction_noisy_batches,
    load_rooom,
)

# ── config ───────────────────────────────────────────────────────────────────
EVAL_DIR   = "/mnt/zone/B/NEW/snpp_eval_loft_230k"
CKPT       = "/mnt/zone/B/NEW/P2P-Bridge-OT-real-latent/checkpoints/PVDL_SNPP_latent/step_230000.pth"
OPT_YAML   = "/mnt/zone/B/NEW/P2P-Bridge-OT-real-latent/checkpoints/PVDL_SNPP_latent/opt.yaml"
GT_SAMPLES = 300_000

FRAME_SIZE = 900        # square render size per frame (px)
POINT_SIZE = 5.0
FRAME_DURATION_MS = 80  # ms per frame in GIF
LOOP_HOLD_MS = 1200     # ms to hold on first (noisy) and last (clean) frame

# ── colormap ─────────────────────────────────────────────────────────────────
_cmap = LinearSegmentedColormap.from_list("err", [
    (0.00, (0.00, 0.55, 0.40)),
    (0.35, (0.35, 0.80, 0.05)),
    (0.55, (0.95, 0.90, 0.00)),
    (0.75, (0.90, 0.30, 0.00)),
    (0.90, (0.80, 0.00, 0.10)),
    (1.00, (0.80, 0.00, 0.60)),
])

def err_colors(pts, gt_pts, vmax_pct=97):
    tree  = cKDTree(gt_pts)
    dists, _ = tree.query(pts, workers=-1)
    vmax  = np.percentile(dists, vmax_pct)
    t     = np.clip(dists / vmax, 0, 1)
    return _cmap(t)[:, :3].astype(np.float32)


# ── renderer ─────────────────────────────────────────────────────────────────
def _mat(pt_size=POINT_SIZE):
    m = rendering.MaterialRecord()
    m.shader     = "defaultUnlit"
    m.point_size = pt_size
    return m

def render_topdown(pts, cols, ref_pts, w=FRAME_SIZE, h=FRAME_SIZE):
    mn, mx  = ref_pts.min(0), ref_pts.max(0)
    center  = (mn + mx) / 2
    ext     = mx - mn
    dist    = max(ext) * 3.0
    eye     = center + np.array([0.0, 0.0, dist])
    aspect  = w / h
    hw      = ext[0] / 2 * 1.10
    hh      = hw / aspect
    if hh < ext[1] / 2 * 1.10:
        hh  = ext[1] / 2 * 1.10
        hw  = hh * aspect

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))

    r = rendering.OffscreenRenderer(w, h)
    r.scene.set_background((1.0, 1.0, 1.0, 1.0))
    r.scene.add_geometry("p", pcd, _mat())
    r.scene.camera.look_at(center.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
    r.scene.camera.set_projection(
        rendering.Camera.Projection.Ortho,
        -hw, hw, -hh, hh, -dist * 2, dist * 2,
    )
    img = np.asarray(r.render_to_image()).copy()
    r.scene.remove_geometry("p")
    del r
    return img


def add_step_label(img_arr, text, size=28, pad=14):
    """Burn a small step label (bottom-left) into the frame."""
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    W, H = img.size
    bb   = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    # white background rect
    draw.rectangle([pad-2, H-th-pad*2, pad+tw+2, H-pad+4], fill=(255,255,255,200))
    draw.text((pad, H-th-pad), text, fill=(20, 20, 20), font=font)
    return np.array(img)


# ── model loading ─────────────────────────────────────────────────────────────
def build_args(scene, steps, gpu="cuda:1"):
    cfg = omegaconf.OmegaConf.load(OPT_YAML)
    overrides = omegaconf.OmegaConf.create({
        "room_path":    os.path.join(EVAL_DIR, scene, "scans", "iphone_dino.ply"),
        "model_path":   CKPT,
        "gpu":          gpu,
        "use_ema":      True,
        "steps":        steps,
        "k":            4,
        "average_predictions": True,
        "batch_size":   16,
        "intermediate": True,
        "seed":         42,
        "feature_name": "dino_iphone",
        "distribution_type": "none",
        "local_rank":   0,
        "restart":      False,
        "out_path":     None,
        "overwrite":    True,
    })
    cfg = omegaconf.OmegaConf.merge(cfg, overrides)
    return cfg


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  default="7bc286c1b6")
    parser.add_argument("--steps",  type=int, default=20,
                        help="Euler ODE steps (= animated frames after noisy)")
    parser.add_argument("--gpu",    default="cuda:1")
    parser.add_argument("--out",    default="/mnt/zone/B/NEW/P2P-Bridge-OT-real-latent/assets/loft-denoise.gif")
    a = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # A5000 — avoid blocking 4090

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Scene   : {a.scene}")
    print(f"Steps   : {a.steps}")
    print(f"Output  : {a.out}")

    args = build_args(a.scene, a.steps, a.gpu)

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    model, _ = load_diffusion(args)

    # ── load room data ────────────────────────────────────────────────────────
    room_path  = args.room_path
    feat_path  = os.path.join(EVAL_DIR, a.scene, "features", "dino_iphone.npy")
    gt_path    = os.path.join(EVAL_DIR, a.scene, "scans", "mesh_aligned_0.05.ply")

    print("Loading room + features …")
    room = o3d.io.read_point_cloud(room_path)
    room_points = np.asarray(room.points)
    room_colors = np.asarray(room.colors) if room.has_colors() else None

    room_dino = np.load(feat_path)   # (384, N) float16
    if room_dino.ndim == 2 and room_dino.shape[0] == 384:
        room_dino = room_dino.T      # (N, 384)
    room_dino = room_dino.astype(np.float32)

    # ── load GT mesh for error coloring ───────────────────────────────────────
    print("Sampling GT mesh …")
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    gt_pts  = np.asarray(gt_mesh.sample_points_uniformly(GT_SAMPLES).points)

    ref_pts = room_points   # shared camera framing

    # ── create patches ────────────────────────────────────────────────────────
    patch_size = args.data.npoints   # 4096
    n_batches  = int(np.ceil(room_points.shape[0] / patch_size) * args.k)

    room_tree  = neighbors.KDTree(room_points, metric="l2")
    pt_torch   = torch.from_numpy(room_points).float().cuda()
    pt_torch   = rearrange(pt_torch, "n d -> 1 d n")
    center_pts = furthest_point_sample(pt_torch, n_batches).squeeze().cpu().numpy().T

    query_radius = 0.3
    idxs_radius  = room_tree.query_radius(center_pts, r=query_radius, return_distance=False)

    print(f"Creating patches (n_batches={n_batches}) …")
    batches_xyz, batches_rgb, batches_dino, batch_idxs, cut_list = create_patches(
        n_batches, room_points, patch_size, idxs_radius, room_colors, room_dino
    )

    batches_xyz   = np.array(batches_xyz)
    batches_dino  = np.array(batches_dino) if len(batches_dino) else None
    batch_idxs    = np.array(batch_idxs)
    cut_idxs      = np.array(cut_list)

    n_minibatch = int(np.ceil(batches_xyz.shape[0] / args.batch_size))
    mb_idxs     = np.array_split(np.arange(batches_xyz.shape[0]), n_minibatch)

    # Accumulation arrays: shape [steps+1, N, 3]
    # Index 0 = noisy, 1..steps = after each Euler step
    N = room_points.shape[0]
    denoised_steps = np.tile(room_points[None], (a.steps + 1, 1, 1)).copy()  # [S+1,N,3]
    n_updates      = np.zeros((a.steps + 1, N), dtype=np.float32)

    # frame 0 = noisy (already set from room_points)
    n_updates[0] = 1.0

    # ── run denoising with step capture ──────────────────────────────────────
    print("Denoising with intermediate steps …")
    for mb in tqdm(mb_idxs, desc="Batches"):
        s, e = mb[0], mb[-1] + 1
        b_xyz  = batches_xyz[s:e]
        b_dino = batches_dino[s:e] if batches_dino is not None else None
        b_idx  = batch_idxs[s:e]
        b_cut  = cut_idxs[s:e]

        patch_final, x_chain = denoise_patch_batch(
            b_xyz, model, args, patch_dino=b_dino, return_steps=True
        )
        # x_chain: [nfe, B, N_pts, 3]  — index 0 = cleanest (REVERSED by sample())
        # Reverse so index 0 = after first step, index nfe-1 = final
        x_chain = x_chain[::-1]   # now [nfe, B, N_pts, 3]: 0=first step, -1=final

        patch_final = np.nan_to_num(patch_final, nan=0.0, posinf=0.0, neginf=0.0)
        x_chain     = np.nan_to_num(x_chain,     nan=0.0, posinf=0.0, neginf=0.0)

        # Accumulate final (step index = a.steps)
        denoised_steps[a.steps], n_updates[a.steps] = update_prediction_noisy_batches(
            denoised_steps[a.steps], n_updates[a.steps], patch_final, b_idx, b_cut
        )

        # Accumulate intermediate steps (1 .. steps-1)
        nfe_available = x_chain.shape[0]   # should equal a.steps
        for t_idx in range(nfe_available):
            frame_idx = t_idx + 1     # 1-based (frame 0 is noisy)
            if frame_idx > a.steps:
                break
            step_pts = x_chain[t_idx]  # [B, N_pts, 3]
            denoised_steps[frame_idx], n_updates[frame_idx] = update_prediction_noisy_batches(
                denoised_steps[frame_idx], n_updates[frame_idx], step_pts, b_idx, b_cut
            )

    # Fill any zero-update points with noisy fallback
    for f in range(1, a.steps + 1):
        unset = n_updates[f] == 0
        if unset.any():
            denoised_steps[f][unset] = room_points[unset]

    # ── render frames ─────────────────────────────────────────────────────────
    print("Rendering frames …")
    frames_pil = []

    # Precompute a shared vmax across ALL frames for consistent color scale
    print("  Precomputing global error scale …")
    all_dists = []
    for f in [0, a.steps // 2, a.steps]:
        pts = denoised_steps[f]
        tree = cKDTree(gt_pts)
        d, _  = tree.query(pts[:5000], workers=-1)   # sample subset for speed
        all_dists.append(d)
    global_vmax = np.percentile(np.concatenate(all_dists), 97)
    print(f"  global_vmax = {global_vmax:.4f}")

    def err_colors_fixed(pts, gt_pts, vmax):
        tree  = cKDTree(gt_pts)
        dists, _ = tree.query(pts, workers=-1)
        t = np.clip(dists / vmax, 0, 1)
        return _cmap(t)[:, :3].astype(np.float32)

    total_frames = a.steps + 1
    for f in tqdm(range(total_frames), desc="Rendering"):
        pts  = denoised_steps[f]
        cols = err_colors_fixed(pts, gt_pts, global_vmax)
        img  = render_topdown(pts, cols, ref_pts)

        # Step label
        if f == 0:
            lbl = "Noisy Input"
        elif f == total_frames - 1:
            lbl = f"Denoised (step {f}/{a.steps})"
        else:
            lbl = f"Step {f}/{a.steps}"
        img = add_step_label(img, lbl)

        frames_pil.append(Image.fromarray(img).convert("P", palette=Image.ADAPTIVE, colors=256))

    # ── assemble GIF ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    durations = [LOOP_HOLD_MS] + [FRAME_DURATION_MS] * (total_frames - 2) + [LOOP_HOLD_MS]

    print(f"Saving GIF ({len(frames_pil)} frames) → {a.out} …")
    frames_pil[0].save(
        a.out,
        save_all=True,
        append_images=frames_pil[1:],
        optimize=False,
        duration=durations,
        loop=0,
    )
    print(f"Done!  {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
