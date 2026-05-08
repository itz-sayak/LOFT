"""
Create a paper-style qualitative montage for ScanNet++ scenes.

Layout:
  rows    = scenes
  columns = [Noisy Input | P2P Original 100k | P2P Original 200k | P2P-Bridge (Ours) | Ground Truth]

Each panel is rendered from the same viewpoint for a scene and colorized by
nearest-neighbor distance to the ground-truth point cloud (green = close,
red/magenta = far). A few black boxes are drawn on the noisy panel and reused
for the remaining columns to mimic the paper-style figure.

Outputs are written to /mnt/zone/B/NEW/snpp_grid/ by default.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


DEFAULT_SCENES = ["7bc286c1b6", "a24f64f7fb", "bcd2436daf"]
DEFAULT_OUT_DIR = "/mnt/zone/B/NEW/snpp_grid"
GT_SAMPLE_POINTS = 500_000
PANEL_W = 920
PANEL_H = 700
LABEL_H = 54
ROW_GAP = 8
BOX_COUNT = 2

NOISY_PATH = "/mnt/zone/B/NEW/snpp_eval_loft_230k/{scene}/scans/iphone_dino.ply"
P2P_100K_PATH = "/mnt/zone/A/P2P_original/snpp_evaluation/{scene}/predictions_dino/P2SB/PVDL-SNPP-gpu0-xyz-20260430_iphone-dino_100000_10_ema.ply"
P2P_200K_PATH = "/mnt/zone/A/P2P_original/snpp_evaluation/{scene}/predictions_dino/P2SB/PVDL-SNPP-latent_iphone-dino_200000_10_ema.ply"
OURS_230K_PATH = "/mnt/zone/B/NEW/snpp_eval_loft_230k/{scene}/predictions_dino/P2SB/PVDL-SNPP-latent_iphone-dino_230000_10_ema.ply"
GT_PATH = "/mnt/zone/B/NEW/snpp_eval_loft_230k/{scene}/scans/mesh_aligned_0.05.ply"

PANEL_SPECS = [
    ("Noisy Input", NOISY_PATH, "pcd"),
    ("P2P Original [100k]", P2P_100K_PATH, "pcd"),
    ("P2P Original [200k]", P2P_200K_PATH, "pcd"),
    ("P2P-Bridge (Ours)", OURS_230K_PATH, "pcd"),
    ("Ground Truth", GT_PATH, "gt"),
]

CMAP = LinearSegmentedColormap.from_list(
    "err",
    [
        (0.00, (0.00, 0.55, 0.40)),  # teal-green
        (0.35, (0.35, 0.80, 0.05)),  # yellow-green
        (0.55, (0.95, 0.90, 0.00)),  # yellow
        (0.75, (0.90, 0.30, 0.00)),  # orange-red
        (0.90, (0.80, 0.00, 0.10)),  # red
        (1.00, (0.80, 0.00, 0.60)),  # magenta
    ],
)


@dataclass
class ScenePanels:
    scene: str
    labels: List[str]
    images: List[Image.Image]


def format_path(template: str, scene: str) -> str:
    return template.format(scene=scene)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")
    points = np.asarray(pcd.points)
    valid = np.isfinite(points).all(axis=1)
    if not valid.all():
        pcd = pcd.select_by_index(np.flatnonzero(valid))
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud has no finite points: {path}")
    return pcd


def load_gt_points(path: str, sample_points: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {path}")
    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    return np.asarray(pcd.points)


def compute_camera(points: np.ndarray, zoom: float = 0.70) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    extent = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    eye = center + np.array([0.45, -0.95, 0.75]) * extent.max() * zoom
    up = np.array([0.0, 0.0, 1.0])
    return eye, center, up


def make_material(point_size: float = 3.5) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    return mat


def render_pcd(
    pcd: o3d.geometry.PointCloud,
    width: int,
    height: int,
    eye: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    point_size: float = 3.5,
) -> np.ndarray:
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))
    renderer.scene.add_geometry("pcd", pcd, make_material(point_size))
    renderer.scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())

    # Keep the same framing across all panels in a scene.
    extent = np.linalg.norm(eye - center)
    renderer.scene.camera.set_projection(
        60.0,
        width / height,
        max(0.05, extent * 0.02),
        extent * 6.0,
        rendering.Camera.FovType.Vertical,
    )

    img = np.asarray(renderer.render_to_image())
    renderer.scene.remove_geometry("pcd")
    del renderer
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def make_error_colors(points: np.ndarray, gt_points: np.ndarray, clip_pct: float = 97.0) -> np.ndarray:
    tree = cKDTree(gt_points)
    dists, _ = tree.query(points, workers=-1)
    vmax = np.percentile(dists, clip_pct)
    vmax = max(vmax, 1e-6)
    t = np.clip(dists / vmax, 0.0, 1.0)
    return CMAP(t)[:, :3].astype(np.float32)


def colorize_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def detect_boxes(img: np.ndarray, box_count: int = 2) -> List[Tuple[int, int, int, int]]:
    """Detect a few dense, high-error windows on the noisy panel."""
    img_f = img.astype(float)
    h, w = img.shape[:2]

    # Background estimate from corners; Open3D/EGL sometimes yields off-white bg.
    c = max(12, min(h, w) // 20)
    corners = np.concatenate(
        [
            img_f[:c, :c].reshape(-1, 3),
            img_f[:c, -c:].reshape(-1, 3),
            img_f[-c:, :c].reshape(-1, 3),
            img_f[-c:, -c:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    fg = np.linalg.norm(img_f - bg[None, None, :], axis=-1) > 18.0

    # Error score: red minus green, but only on foreground pixels.
    err = np.clip(img_f[:, :, 0] - img_f[:, :, 1], 0.0, 255.0) * fg.astype(float)
    score_map = err * fg.astype(float)

    box_h = max(120, int(h * 0.22))
    box_w = max(120, int(w * 0.22))
    step_y = max(24, box_h // 3)
    step_x = max(24, box_w // 3)

    boxes: List[Tuple[int, int, int, int]] = []
    for _ in range(box_count):
        best_score = -1.0
        best_box = None
        for y in range(0, max(1, h - box_h), step_y):
            for x in range(0, max(1, w - box_w), step_x):
                reg_fg = fg[y : y + box_h, x : x + box_w]
                if reg_fg.mean() < 0.25:
                    continue
                score = score_map[y : y + box_h, x : x + box_w].mean()
                if score > best_score:
                    best_score = score
                    best_box = (x, y, x + box_w, y + box_h)
        if best_box is None:
            break
        boxes.append(best_box)
        x0, y0, x1, y1 = best_box
        pad = box_h // 2
        score_map[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)] = 0.0
    return boxes


def draw_boxes(img: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for x0, y0, x1, y1 in boxes:
        for t in range(3):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=(0, 0, 0))
    return pil


def label_bar(width: int, text: str) -> Image.Image:
    bar = Image.new("RGB", (width, LABEL_H), (255, 255, 255))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((width - tw) // 2, (LABEL_H - th) // 2), text, fill=(0, 0, 0), font=font)
    return bar


def compose_row(images: Sequence[Image.Image]) -> Image.Image:
    return Image.fromarray(np.concatenate([np.asarray(im) for im in images], axis=1))


def render_scene(scene: str) -> ScenePanels:
    noisy_path = format_path(NOISY_PATH, scene)
    gt_path = format_path(GT_PATH, scene)

    gt_points = load_gt_points(gt_path, GT_SAMPLE_POINTS)
    noisy_pcd = load_point_cloud(noisy_path)
    noisy_points = np.asarray(noisy_pcd.points)
    eye, center, up = compute_camera(noisy_points)

    panel_defs: List[Tuple[str, str, str]] = []
    for label, template, kind in PANEL_SPECS:
        panel_defs.append((label, format_path(template, scene), kind))

    # Keep the same GT sample for all panels in a scene.
    gt_colors = np.tile(np.array([[0.0, 0.55, 0.40]], dtype=np.float32), (len(gt_points), 1))
    gt_render = colorize_point_cloud(gt_points, gt_colors)

    rendered: List[np.ndarray] = []
    labels: List[str] = []

    # We color all non-GT panels by distance to GT.
    for label, path, kind in panel_defs:
        labels.append(label)
        if kind == "gt":
            pcd = gt_render
        else:
            pcd_raw = load_point_cloud(path)
            points = np.asarray(pcd_raw.points)
            colors = make_error_colors(points, gt_points)
            pcd = colorize_point_cloud(points, colors)
        rendered.append(render_pcd(pcd, PANEL_W, PANEL_H, eye, center, up))

    boxes = detect_boxes(rendered[0], BOX_COUNT)
    boxed_images = [draw_boxes(img, boxes) for img in rendered]
    return ScenePanels(scene=scene, labels=labels, images=boxed_images)


def save_scene_row(scene_panels: ScenePanels, out_dir: Path) -> Path:
    scene_dir = out_dir / scene_panels.scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    row_img = compose_row(scene_panels.images)
    row_img.save(scene_dir / "row.png")

    # Save the individual panels too, in case the user wants them separately.
    for label, image in zip(scene_panels.labels, scene_panels.images):
        safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        image.save(scene_dir / f"{safe}.png")

    return scene_dir / "row.png"


def compose_grid(row_paths: Sequence[Path], labels: Sequence[str], out_path: Path) -> None:
    row_images = [Image.open(path).convert("RGB") for path in row_paths]
    widths = {img.size[0] for img in row_images}
    heights = {img.size[1] for img in row_images}
    if len(widths) != 1 or len(heights) != 1:
        raise RuntimeError(f"Row images have mismatched sizes: widths={widths}, heights={heights}")

    panel_w = row_images[0].size[0] // len(labels)
    row_w = row_images[0].size[0]
    row_h = row_images[0].size[1]
    total_w = row_w
    total_h = len(row_images) * row_h + LABEL_H + ROW_GAP * (len(row_images) - 1)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    y = 0
    for img in row_images:
        canvas.paste(img, (0, y))
        y += row_h + ROW_GAP

    # Column labels at the bottom.
    label_strip = Image.new("RGB", (total_w, LABEL_H), (255, 255, 255))
    draw = ImageDraw.Draw(label_strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        x0 = i * panel_w
        x1 = x0 + panel_w
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((x0 + (panel_w - tw) // 2, (LABEL_H - th) // 2), label, fill=(0, 0, 0), font=font)
    canvas.paste(label_strip, (0, total_h - LABEL_H))
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-style ScanNet++ qualitative montage.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for the montage.")
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES, help="Scene IDs to render.")
    parser.add_argument("--include-5748", action="store_true", help="Include the outlier scene 5748ce6f01.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = list(args.scenes)
    if args.include_5748 and "5748ce6f01" not in scenes:
        scenes.append("5748ce6f01")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_paths: List[Path] = []
    labels: List[str] = []
    for scene in scenes:
        print(f"\n=== {scene} ===")
        scene_panels = render_scene(scene)
        row_path = save_scene_row(scene_panels, out_dir)
        row_paths.append(row_path)
        labels = scene_panels.labels
        print(f"  saved: {row_path}")

    grid_path = out_dir / "comparison_grid.png"
    compose_grid(row_paths, labels, grid_path)
    print(f"\nGrid saved to {grid_path}")


if __name__ == "__main__":
    main()
