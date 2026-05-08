"""
Paper-style 4-column qualitative comparison for ScanNet++.

Columns:  Noisy Input | P2P-Bridge | LOFT (Ours) | Ground Truth (Faro)
Rows   :  one per scene
Color  :  distance to GT  →  teal-green (close) → yellow → red → magenta (far)
View   :  orthographic top-down

Output:
  /mnt/zone/B/NEW/snpp_vis3/{scene}/noisy.png
  /mnt/zone/B/NEW/snpp_vis3/{scene}/p2p_bridge.png
  /mnt/zone/B/NEW/snpp_vis3/{scene}/loft.png
  /mnt/zone/B/NEW/snpp_vis3/{scene}/gt.png
  /mnt/zone/B/NEW/snpp_vis3/{scene}/row.png          ← horizontal strip for that scene
  /mnt/zone/B/NEW/snpp_vis3/montage.png              ← all 3 scenes stacked vertically
"""

import os
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

# ── paths ─────────────────────────────────────────────────────────────────────
LOFT_DIR   = "/mnt/zone/B/NEW/snpp_eval_loft_230k"
P2P_DIR    = "/mnt/zone/A/P2P_original/snpp_evaluation"
OUT_DIR    = "/mnt/zone/B/NEW/snpp_vis3"
SCENES     = ["7bc286c1b6", "a24f64f7fb", "bcd2436daf"]

LOFT_PLY   = "PVDL-SNPP-latent_iphone-dino_230000_10_ema.ply"
P2P_PLY    = "PVDL-SNPP-gpu0-xyz-20260430_iphone-dino_100000_10_ema.ply"

FONT_PATH  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ── render config ─────────────────────────────────────────────────────────────
W, H        = 900, 780    # per-panel resolution
PT_SIZE     = 5.0
GT_SAMPLES  = 500_000
GAP         = 6           # pixel gap between panels
LABEL_H     = 46          # label strip height below each row

# ── error colormap: teal-green → yellow → red → magenta ──────────────────────
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

def sanitize(pts):
    """Remove non-finite rows."""
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]

# ── rendering ─────────────────────────────────────────────────────────────────
def _mat(pt_size=PT_SIZE):
    m = rendering.MaterialRecord()
    m.shader    = "defaultUnlit"
    m.point_size = pt_size
    return m

def render_topdown(pts, cols, ref_pts, w=W, h=H):
    """Orthographic top-down render of pts with per-point cols (float32 RGB)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))

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

# ── auto-detect background + foreground mask ──────────────────────────────────
def fg_mask(img_arr, corner=12, tol=18):
    f = img_arr.astype(float)
    bg = np.concatenate([
        f[:corner, :corner].reshape(-1,3),
        f[:corner,-corner:].reshape(-1,3),
        f[-corner:,:corner].reshape(-1,3),
        f[-corner:,-corner:].reshape(-1,3),
    ]).mean(0)
    return np.abs(f - bg).max(-1) > tol

# ── inset box detection (main room body only) ─────────────────────────────────
def find_boxes(img_arr, n=2, frac=0.22, min_dens=0.40, row_thr=0.25):
    fg   = fg_mask(img_arr)
    ih, iw = img_arr.shape[:2]
    ch_r, ch_g = img_arr[:,:,0].astype(float), img_arr[:,:,1].astype(float)
    err  = np.clip(ch_r - ch_g, 0, 255) * fg

    # restrict to rows / cols with dense foreground coverage
    yd = np.where(fg.mean(1) > row_thr)[0]
    xd = np.where(fg.mean(0) > row_thr)[0]
    pad = int(ih * 0.02)
    ys = max(0, yd[0] - pad)   if len(yd) else 0
    ye = min(ih, yd[-1] + pad) if len(yd) else ih
    xs = max(0, xd[0] - pad)   if len(xd) else 0
    xe = min(iw, xd[-1] + pad) if len(xd) else iw

    bh, bw    = int(ih * frac), int(iw * frac)
    sy, sx    = max(1, bh // 3), max(1, bw // 3)
    score_map = err * fg

    boxes = []
    for _ in range(n):
        best, box = -1, None
        for y in range(ys, ye - bh, sy):
            for x in range(xs, xe - bw, sx):
                if fg[y:y+bh, x:x+bw].mean() < min_dens:
                    continue
                s = score_map[y:y+bh, x:x+bw].mean()
                if s > best:
                    best, box = s, (x, y, x+bw, y+bh)
        if box is None:
            break
        boxes.append(box)
        x0,y0,x1,y1 = box
        p2 = bh // 2
        score_map[max(0,y0-p2):min(ih,y1+p2), max(0,x0-p2):min(iw,x1+p2)] = 0
    return boxes

# ── apply boxes to a PIL image (inset crops at bottom corners) ────────────────
def apply_boxes(img_pil, boxes, inset_frac=0.30, brd=3):
    img  = img_pil.copy()
    draw = ImageDraw.Draw(img)
    IW, IH = img.size
    iw_box = int(IW * inset_frac)
    anchors = [(4, IH), (IW - iw_box - 4 - 2*brd, IH)]

    for i, (x0,y0,x1,y1) in enumerate(boxes):
        for t in range(brd):
            draw.rectangle([x0-t, y0-t, x1+t, y1+t], outline=(0,0,0))
        crop   = img_pil.crop((x0,y0,x1,y1))
        bw_,bh_ = x1-x0, y1-y0
        ih_box = int(iw_box * bh_ / bw_)
        zoomed = crop.resize((iw_box, ih_box), Image.LANCZOS)
        framed = Image.new("RGB", (iw_box+2*brd, ih_box+2*brd), (0,0,0))
        framed.paste(zoomed, (brd, brd))
        ax, ay_base = anchors[i % len(anchors)]
        ay = ay_base - framed.height - 4
        if ay < 0: ay = 4
        img.paste(framed, (ax, ay))
    return img

# ── label strip ───────────────────────────────────────────────────────────────
def label_strip(width, labels, bg=(245,245,245), fg_col=(20,20,20)):
    """One horizontal label strip with evenly-spaced column names."""
    bar  = Image.new("RGB", (width, LABEL_H), bg)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype(FONT_PATH, 22)
    except Exception:
        font = ImageFont.load_default()
    n  = len(labels)
    cw = width // n
    for i, text in enumerate(labels):
        bb = draw.textbbox((0,0), text, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        x = i*cw + (cw - tw)//2
        y = (LABEL_H - th)//2
        draw.text((x, y), text, fill=fg_col, font=font)
    return np.array(bar)

# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
COL_NAMES = ["Noisy Input", "P2P-Bridge", "LOFT (Ours)", "Ground Truth (Faro)"]
BG_COLOR  = np.array([255, 255, 255], dtype=np.uint8)

all_rows = []   # collect per-scene rows for final montage

for scene in SCENES:
    print(f"\n{'='*52}\n  Scene: {scene}\n{'='*52}")
    scene_dir = os.path.join(OUT_DIR, scene)
    os.makedirs(scene_dir, exist_ok=True)

    # ── file paths ────────────────────────────────────────────────────────────
    noisy_path = os.path.join(LOFT_DIR, scene, "scans", "iphone_dino.ply")
    loft_path  = os.path.join(LOFT_DIR, scene, "predictions_dino", "P2SB", LOFT_PLY)
    p2p_path   = os.path.join(P2P_DIR,  scene, "predictions_dino", "P2SB", P2P_PLY)
    gt_path    = os.path.join(LOFT_DIR, scene, "scans", "mesh_aligned_0.05.ply")

    # ── load data ─────────────────────────────────────────────────────────────
    print("  Loading GT mesh …")
    gt_mesh  = o3d.io.read_triangle_mesh(gt_path)
    gt_sampled = gt_mesh.sample_points_uniformly(number_of_points=GT_SAMPLES)
    gt_pts   = sanitize(np.asarray(gt_sampled.points))

    noisy_pts = sanitize(np.asarray(o3d.io.read_point_cloud(noisy_path).points))
    loft_pts  = sanitize(np.asarray(o3d.io.read_point_cloud(loft_path).points))
    p2p_pts   = sanitize(np.asarray(o3d.io.read_point_cloud(p2p_path).points))

    ref_pts = noisy_pts   # shared camera framing

    # ── compute error colors vs GT ────────────────────────────────────────────
    print("  Computing error colors …")
    noisy_col = err_colors(noisy_pts, gt_pts)
    loft_col  = err_colors(loft_pts,  gt_pts)
    p2p_col   = err_colors(p2p_pts,   gt_pts)
    gt_col    = np.tile([0.00, 0.55, 0.40], (len(gt_pts), 1)).astype(np.float32)

    # ── render all 4 panels ───────────────────────────────────────────────────
    print("  Rendering …", end=" ", flush=True)
    panels_arr = {}
    for name, pts, col in [
        ("noisy",     noisy_pts, noisy_col),
        ("p2p_bridge",p2p_pts,   p2p_col),
        ("loft",      loft_pts,  loft_col),
        ("gt",        gt_pts,    gt_col),
    ]:
        print(name, end=" ", flush=True)
        panels_arr[name] = render_topdown(pts, col, ref_pts)
    print()

    # ── find inset boxes on noisy panel (shared across all columns) ───────────
    boxes = find_boxes(panels_arr["noisy"])
    print(f"  Crop boxes: {boxes}")

    # ── apply boxes & save individual images ──────────────────────────────────
    panels_pil = {}
    for name, arr in panels_arr.items():
        p = apply_boxes(Image.fromarray(arr), boxes)
        panels_pil[name] = p
        p.save(os.path.join(scene_dir, f"{name}.png"))

    # ── build horizontal row (panels side by side, label strip below) ─────────
    order = ["noisy", "p2p_bridge", "loft", "gt"]
    gap_col = np.full((H, GAP, 3), 220, dtype=np.uint8)   # light-grey gap
    row_panels = []
    for i, name in enumerate(order):
        row_panels.append(np.array(panels_pil[name]))
        if i < len(order) - 1:
            row_panels.append(gap_col)
    row_img  = np.concatenate(row_panels, axis=1)

    # label strip spans full row width
    row_w    = row_img.shape[1]
    lbl      = label_strip(row_w, COL_NAMES)
    row_full = np.concatenate([row_img, lbl], axis=0)

    Image.fromarray(row_full).save(os.path.join(scene_dir, "row.png"))
    all_rows.append(row_full)
    print(f"  → {scene_dir}/row.png  ({row_full.shape[1]}×{row_full.shape[0]})")

# ── stack all scene rows into final montage ───────────────────────────────────
print("\nBuilding final montage …")
h_sep = np.full((10, all_rows[0].shape[1], 3), 210, dtype=np.uint8)
parts = []
for i, row in enumerate(all_rows):
    parts.append(row)
    if i < len(all_rows) - 1:
        parts.append(h_sep)

montage = np.concatenate(parts, axis=0)
out_path = os.path.join(OUT_DIR, "montage.png")
Image.fromarray(montage).save(out_path)
print(f"\nDone!  {out_path}  ({montage.shape[1]}×{montage.shape[0]} px)")
