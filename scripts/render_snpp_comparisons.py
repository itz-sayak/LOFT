"""
Render side-by-side comparison images for ScanNet++ scenes:
  noisy input  |  denoised output (LOFT 230k)  |  ground truth mesh (optional)

Output: /mnt/zone/B/NEW/snpp_vis/{scene}/noisy.png
        /mnt/zone/B/NEW/snpp_vis/{scene}/denoised.png
        /mnt/zone/B/NEW/snpp_vis/{scene}/comparison.png
"""

import os
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
EVAL_DIR = "/mnt/zone/B/NEW/snpp_eval_loft_230k"
OUT_DIR  = "/mnt/zone/B/NEW/snpp_vis"
SCENES   = ["7bc286c1b6", "a24f64f7fb", "bcd2436daf"]   # skip 5748 (outlier)
PRED_PLY = "PVDL-SNPP-latent_iphone-dino_230000_10_ema.ply"

W, H = 1280, 960          # render resolution per panel
POINT_SIZE = 2.0          # point size in pixels
BG_COLOR   = (0.12, 0.12, 0.12, 1.0)   # dark background


def chamfer_color_map(points_pred, points_gt, clip_pct=99):
    """Color pred points by nearest-neighbor distance to GT (red=far, green=close)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points_gt)
    dists, _ = tree.query(points_pred, workers=-1)
    vmax = np.percentile(dists, clip_pct)
    dists_norm = np.clip(dists / vmax, 0, 1)
    # red -> yellow -> green  (matplotlib RdYlGn reversed)
    colors = np.zeros((len(dists_norm), 3), dtype=np.float32)
    # simple 3-stop gradient: green(0) -> yellow(0.5) -> red(1)
    colors[:, 0] = np.clip(2 * dists_norm, 0, 1)        # R
    colors[:, 1] = np.clip(2 * (1 - dists_norm), 0, 1)  # G
    colors[:, 2] = 0.0                                    # B
    return colors


def make_material(point_size=POINT_SIZE):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    return mat


def compute_camera(points, zoom=0.65):
    """Return eye, center, up for a nice isometric-ish view of points."""
    center = points.mean(axis=0)
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    eye = center + np.array([0.4, -0.9, 0.8]) * extent * zoom
    up  = np.array([0.0, 0.0, 1.0])
    return eye, center, up


def render_pcd(pcd, width=W, height=H, eye=None, center=None, up=None):
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background(BG_COLOR)
    r.scene.add_geometry("pcd", pcd, make_material())

    # fit to scene bounds if no camera given
    bounds = r.scene.bounding_box
    if eye is None:
        r.setup_camera(60.0, bounds, bounds.get_center())
    else:
        r.scene.camera.look_at(center, eye, up)

    img = r.render_to_image()
    r.scene.remove_geometry("pcd")
    del r
    return np.asarray(img)


def make_label_bar(width, text, bg=(30, 30, 30), fg=(240, 240, 240)):
    """Create a narrow banner with centered text using PIL."""
    from PIL import ImageDraw, ImageFont
    bar = Image.new("RGB", (width, 48), bg)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((width - tw) // 2, (48 - th) // 2), text, fill=fg, font=font)
    return np.array(bar)


os.makedirs(OUT_DIR, exist_ok=True)

for scene in SCENES:
    print(f"\n=== {scene} ===")
    noisy_path = os.path.join(EVAL_DIR, scene, "scans", "iphone_dino.ply")
    pred_path  = os.path.join(EVAL_DIR, scene, "predictions_dino", "P2SB", PRED_PLY)

    noisy_pcd = o3d.io.read_point_cloud(noisy_path)
    pred_pcd  = o3d.io.read_point_cloud(pred_path)

    noisy_pts = np.asarray(noisy_pcd.points)
    pred_pts  = np.asarray(pred_pcd.points)

    # shared camera based on noisy point extents
    eye, center, up = compute_camera(noisy_pts)

    # ── 1. noisy input with Chamfer-error coloring vs. prediction ─────────────
    noisy_err_colors = chamfer_color_map(noisy_pts, pred_pts)
    noisy_vis = o3d.geometry.PointCloud()
    noisy_vis.points = o3d.utility.Vector3dVector(noisy_pts)
    noisy_vis.colors = o3d.utility.Vector3dVector(noisy_err_colors)

    # ── 2. denoised prediction with Chamfer-error coloring vs. noisy ──────────
    pred_err_colors = chamfer_color_map(pred_pts, noisy_pts)
    pred_vis = o3d.geometry.PointCloud()
    pred_vis.points = o3d.utility.Vector3dVector(pred_pts)
    pred_vis.colors = o3d.utility.Vector3dVector(pred_err_colors)

    print("  Rendering noisy …")
    img_noisy  = render_pcd(noisy_vis, eye=eye, center=center, up=up)
    print("  Rendering denoised …")
    img_pred   = render_pcd(pred_vis,  eye=eye, center=center, up=up)

    # ── also render with original RGB colors ──────────────────────────────────
    print("  Rendering noisy RGB …")
    img_noisy_rgb = render_pcd(noisy_pcd, eye=eye, center=center, up=up)
    print("  Rendering denoised RGB …")
    img_pred_rgb  = render_pcd(pred_pcd,  eye=eye, center=center, up=up)

    scene_dir = os.path.join(OUT_DIR, scene)
    os.makedirs(scene_dir, exist_ok=True)

    # save individual images
    Image.fromarray(img_noisy_rgb).save(os.path.join(scene_dir, "noisy_rgb.png"))
    Image.fromarray(img_pred_rgb).save(os.path.join(scene_dir, "denoised_rgb.png"))
    Image.fromarray(img_noisy).save(os.path.join(scene_dir, "noisy_error.png"))
    Image.fromarray(img_pred).save(os.path.join(scene_dir, "denoised_error.png"))

    # ── compose side-by-side comparison ───────────────────────────────────────
    label_noisy    = make_label_bar(W, "Noisy Input")
    label_denoised = make_label_bar(W, "Denoised (LOFT 230k)")
    label_noisy_e  = make_label_bar(W, "Noisy Input (error map)")
    label_pred_e   = make_label_bar(W, "Denoised (error map)")

    bar_h = label_noisy.shape[0]
    panel_h = H + bar_h

    # top row: RGB colors
    top_row = np.concatenate([
        np.concatenate([label_noisy, img_noisy_rgb], axis=0),
        np.concatenate([label_denoised, img_pred_rgb], axis=0),
    ], axis=1)

    # bottom row: error maps
    bot_row = np.concatenate([
        np.concatenate([label_noisy_e, img_noisy], axis=0),
        np.concatenate([label_pred_e,  img_pred], axis=0),
    ], axis=1)

    comparison = np.concatenate([top_row, bot_row], axis=0)
    Image.fromarray(comparison).save(os.path.join(scene_dir, "comparison.png"))
    print(f"  Saved to {scene_dir}/")

print("\nDone! All images at", OUT_DIR)
