"""
Render paper-style top-down 2D visualizations of ScanNet++ point clouds.

Color scheme: green (close to GT) → yellow → red → magenta (far from GT)
View: orthographic top-down (Z-up, looking along -Z axis)
Extras: automatic inset crop boxes on high-error regions

Output layout per scene:
  snpp_vis2/{scene}/noisy.png
  snpp_vis2/{scene}/denoised.png
  snpp_vis2/{scene}/gt.png
  snpp_vis2/{scene}/comparison.png  ← side-by-side strip
"""

import os
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

# ── config ────────────────────────────────────────────────────────────────────
EVAL_DIR = "/mnt/zone/B/NEW/snpp_eval_loft_230k"
OUT_DIR  = "/mnt/zone/B/NEW/snpp_vis2"
SCENES   = ["7bc286c1b6", "a24f64f7fb", "bcd2436daf"]
PRED_PLY = "PVDL-SNPP-latent_iphone-dino_230000_10_ema.ply"

W, H       = 1200, 900    # render resolution per panel
POINT_SIZE = 5.0
GT_SAMPLES = 500_000      # points to sample from GT mesh for distance queries

# ── custom colormap: teal-green → yellow → red → magenta ─────────────────────
_cmap = LinearSegmentedColormap.from_list("err", [
    (0.00, (0.00, 0.55, 0.40)),   # teal-green  (low error)
    (0.35, (0.35, 0.80, 0.05)),   # yellow-green
    (0.55, (0.95, 0.90, 0.00)),   # yellow
    (0.75, (0.90, 0.30, 0.00)),   # orange-red
    (0.90, (0.80, 0.00, 0.10)),   # red
    (1.00, (0.80, 0.00, 0.60)),   # magenta      (high error)
])

def error_colors(points, gt_pts, vmax_pct=97):
    """Color `points` by nearest-neighbour distance to `gt_pts`."""
    tree = cKDTree(gt_pts)
    dists, _ = tree.query(points, workers=-1)
    vmax = np.percentile(dists, vmax_pct)
    t = np.clip(dists / vmax, 0, 1)
    return _cmap(t)[:, :3].astype(np.float32)


# ── rendering ─────────────────────────────────────────────────────────────────
def _make_mat(pt_size=POINT_SIZE):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = pt_size
    return mat


def render_topdown(pcd, ref_pts, w=W, h=H, pt_size=POINT_SIZE):
    """
    Render `pcd` orthographically from directly above (Z-up).
    `ref_pts` is used to define scene bounds / camera framing.
    Returns an (H, W, 3) uint8 numpy array.
    """
    pts_min = ref_pts.min(0)
    pts_max = ref_pts.max(0)
    center  = (pts_min + pts_max) / 2
    extent  = pts_max - pts_min       # [ex, ey, ez]

    # Eye directly above center
    dist = max(extent) * 3.0
    eye  = center + np.array([0.0, 0.0, dist])
    up   = np.array([0.0, 1.0, 0.0])

    # Orthographic bounds that fit XY extent
    aspect = w / h
    half_x = extent[0] / 2 * 1.08
    half_y = extent[1] / 2 * 1.08
    if half_x / half_y > aspect:
        hw = half_x
        hh = hw / aspect
    else:
        hh = half_y
        hw = hh * aspect

    r = rendering.OffscreenRenderer(w, h)
    r.scene.set_background((1.0, 1.0, 1.0, 1.0))
    r.scene.add_geometry("pcd", pcd, _make_mat(pt_size))

    r.scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())
    r.scene.camera.set_projection(
        rendering.Camera.Projection.Ortho,
        -hw, hw,        # left, right  (camera-space X)
        -hh, hh,        # bottom, top  (camera-space Y)
        -dist * 2,      # near
         dist * 2,      # far
    )

    img = np.asarray(r.render_to_image())
    r.scene.remove_geometry("pcd")
    del r
    return img


# ── inset crop boxes ──────────────────────────────────────────────────────────
def find_crop_boxes(img_arr, n=2, box_frac=0.22, min_fg_frac=0.40,
                    row_density_thresh=0.25):
    """
    Find n interesting (high-error / reddish) crop regions in a white-bg image.
    Auto-detects background from corner samples, then restricts search to the
    main room body (rows with high foreground density) to avoid isolated outliers.
    Returns list of (x0, y0, x1, y1) pixel boxes.
    """
    from scipy.ndimage import binary_erosion

    img_f = img_arr.astype(float)
    # auto-detect background color from 4 corners
    c = 15  # corner sample size
    bg = np.concatenate([
        img_f[:c,  :c].reshape(-1, 3),
        img_f[:c, -c:].reshape(-1, 3),
        img_f[-c:, :c].reshape(-1, 3),
        img_f[-c:,-c:].reshape(-1, 3),
    ]).mean(axis=0)

    # foreground: any channel differs from bg by more than 18
    diff = np.abs(img_f - bg).max(axis=-1)
    fg   = diff > 18

    ih, iw = img_arr.shape[:2]
    ch_r, ch_g = img_f[:, :, 0], img_f[:, :, 1]
    err = np.clip(ch_r - ch_g, 0, 255) * fg.astype(float)

    # ── restrict to main room body using row-density profile ─────────────────
    row_dens = fg.mean(axis=1)  # fraction of foreground pixels per row
    in_room  = row_dens > row_density_thresh
    main_rows = np.where(in_room)[0]
    pad = int(ih * 0.02)
    if len(main_rows) >= 2:
        ys = max(0, main_rows[0] - pad)
        ye = min(ih, main_rows[-1] + pad)
    else:
        ys, ye = 0, ih

    # similarly restrict columns
    col_dens  = fg.mean(axis=0)
    in_room_c = col_dens > row_density_thresh
    main_cols = np.where(in_room_c)[0]
    if len(main_cols) >= 2:
        xs = max(0, main_cols[0] - pad)
        xe = min(iw, main_cols[-1] + pad)
    else:
        xs, xe = 0, iw

    bh, bw  = int(ih * box_frac), int(iw * box_frac)
    step_y  = max(1, bh // 3)
    step_x  = max(1, bw // 3)

    # score = error × density (favours dense AND noisy)
    score_map = err * fg.astype(float)

    boxes = []
    for _ in range(n):
        best_score, best_box = -1, None
        for y in range(ys, ye - bh, step_y):
            for x in range(xs, xe - bw, step_x):
                reg_fg = fg[y:y+bh, x:x+bw]
                if reg_fg.mean() < min_fg_frac:
                    continue
                s = score_map[y:y+bh, x:x+bw].mean()
                if s > best_score:
                    best_score = s
                    best_box = (x, y, x + bw, y + bh)
        if best_box is None:
            break
        boxes.append(best_box)
        x0, y0, x1, y1 = best_box
        pad2 = bh // 2
        score_map[max(0, y0-pad2):min(ih, y1+pad2),
                  max(0, x0-pad2):min(iw, x1+pad2)] = 0
    return boxes


def add_insets(img_pil, boxes, inset_frac=0.30, border=3):
    """
    Draw black crop-box outlines on `img_pil` and paste zoomed insets
    in the bottom corners.  Returns a new PIL Image.
    """
    img  = img_pil.copy()
    draw = ImageDraw.Draw(img)
    IW, IH = img.size
    inset_w = int(IW * inset_frac)

    # Two possible anchor positions: bottom-left, bottom-right
    anchors = [(5, IH), (IW - inset_w - 5 - 2 * border, IH)]

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0

        # draw rectangle (thick)
        for t in range(border):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=(0, 0, 0))

        # crop + zoom
        crop   = img_pil.crop((x0, y0, x1, y1))
        inh    = int(inset_w * bh / bw)
        zoomed = crop.resize((inset_w, inh), Image.LANCZOS)

        # black-bordered frame
        framed = Image.new("RGB", (inset_w + 2 * border, inh + 2 * border), (0, 0, 0))
        framed.paste(zoomed, (border, border))

        # anchor: bottom of image, left or right side
        ax, ay_base = anchors[i % len(anchors)]
        ay = ay_base - framed.height - 5
        if ay < 0:
            ay = 5
        img.paste(framed, (ax, ay))

    return img


def label_bar(width, text, bg=(30, 30, 30), fg=(240, 240, 240)):
    bar  = Image.new("RGB", (width, 52), bg)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    bb   = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((width - tw) // 2, (52 - th) // 2), text, fill=fg, font=font)
    return np.array(bar)


# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for scene in SCENES:
    print(f"\n{'='*50}\n  Scene: {scene}\n{'='*50}")
    noisy_path = os.path.join(EVAL_DIR, scene, "scans", "iphone_dino.ply")
    pred_path  = os.path.join(EVAL_DIR, scene, "predictions_dino", "P2SB", PRED_PLY)
    gt_path    = os.path.join(EVAL_DIR, scene, "scans", "mesh_aligned_0.05.ply")

    # ── load GT mesh → sample dense point cloud for distance queries ──────────
    print("  Sampling GT mesh …")
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    gt_pcd  = gt_mesh.sample_points_uniformly(number_of_points=GT_SAMPLES)
    gt_pts  = np.asarray(gt_pcd.points)

    # ── load input / prediction ───────────────────────────────────────────────
    noisy_pcd = o3d.io.read_point_cloud(noisy_path)
    pred_pcd  = o3d.io.read_point_cloud(pred_path)
    noisy_pts = np.asarray(noisy_pcd.points)
    pred_pts  = np.asarray(pred_pcd.points)

    # ── error-color all three ─────────────────────────────────────────────────
    print("  Computing error colors …")
    noisy_col = error_colors(noisy_pts, gt_pts)
    pred_col  = error_colors(pred_pts,  gt_pts)
    # GT shown in uniform teal-green (it IS the reference)
    gt_col    = np.tile([0.0, 0.55, 0.40], (len(gt_pts), 1)).astype(np.float32)

    def make_pcd(pts, col):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p.colors = o3d.utility.Vector3dVector(col)
        return p

    noisy_vis = make_pcd(noisy_pts, noisy_col)
    pred_vis  = make_pcd(pred_pts,  pred_col)
    gt_vis    = make_pcd(gt_pts,    gt_col)

    # Camera framing based on noisy input extent (shared across all panels)
    ref_pts = noisy_pts

    # ── render all panels ─────────────────────────────────────────────────────
    print("  Rendering noisy …")
    img_noisy_arr = render_topdown(noisy_vis, ref_pts)
    print("  Rendering denoised …")
    img_pred_arr  = render_topdown(pred_vis,  ref_pts)
    print("  Rendering GT …")
    img_gt_arr    = render_topdown(gt_vis,    ref_pts)

    # ── find inset boxes on noisy (highest-error = most interesting) ──────────
    boxes = find_crop_boxes(img_noisy_arr, n=2)
    print(f"  Crop boxes: {boxes}")

    # ── apply insets to all panels ────────────────────────────────────────────
    panels = {}
    for name, arr in [("noisy", img_noisy_arr),
                      ("denoised", img_pred_arr),
                      ("gt", img_gt_arr)]:
        pil = Image.fromarray(arr)
        if boxes:
            pil = add_insets(pil, boxes)
        panels[name] = pil

    # ── save individual images ────────────────────────────────────────────────
    scene_dir = os.path.join(OUT_DIR, scene)
    os.makedirs(scene_dir, exist_ok=True)

    for name, pil in panels.items():
        pil.save(os.path.join(scene_dir, f"{name}.png"))

    # ── compose side-by-side comparison strip ─────────────────────────────────
    labels = {
        "noisy":    "Noisy Input",
        "denoised": "Denoised (LOFT 230k)",
        "gt":       "Ground Truth (Faro)",
    }
    strip_parts = []
    for name in ["noisy", "denoised", "gt"]:
        bar   = label_bar(W, labels[name])
        panel = np.array(panels[name])
        strip_parts.append(np.concatenate([bar, panel], axis=0))

    comparison = np.concatenate(strip_parts, axis=1)
    Image.fromarray(comparison).save(os.path.join(scene_dir, "comparison.png"))
    print(f"  → saved to {scene_dir}/")

print(f"\nDone! All images at {OUT_DIR}")
