import numpy as np
import open3d as o3d
import os

MAX_POINTS = 500000
SEED = 42

for scene in ["7bc286c1b6", "a24f64f7fb", "bcd2436daf", "5748ce6f01"]:
    if scene == "5748ce6f01":
        src_ply = f"/mnt/zone/B/NEW/P2B_latent_DDPM/data/scannetpp_release/data/{scene}/scans/iphone.ply"
    else:
        src_ply = f"/mnt/zone/B/NEW/P2P original/snpp_evaluation_gpu0_xyz_20260430_3scenes/{scene}/scans/iphone.ply"
    out_ply = f"/mnt/zone/B/NEW/snpp_eval_loft_230k/{scene}/scans/iphone_dino.ply"

    pcd = o3d.io.read_point_cloud(src_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # same logic as P2B_latent_DDPM/data/processing/image_features.py
    removal_mask = np.any(np.isnan(points), axis=1) | np.any(np.isinf(points), axis=1)
    points = points[~removal_mask]
    colors = colors[~removal_mask]

    print(f"{scene}: {len(points)} points in iphone.ply", flush=True)

    if len(points) > MAX_POINTS:
        rng = np.random.default_rng(seed=SEED)
        idx = rng.choice(len(points), size=MAX_POINTS, replace=False)
        idx.sort()
        points = points[idx]
        colors = colors[idx]

    pcd_ds = o3d.geometry.PointCloud()
    pcd_ds.points = o3d.utility.Vector3dVector(points)
    if len(colors) > 0:
        pcd_ds.colors = o3d.utility.Vector3dVector(colors)

    # Replace the symlink with a real file
    if os.path.islink(out_ply):
        os.unlink(out_ply)
    o3d.io.write_point_cloud(out_ply, pcd_ds)
    print(f"  -> saved {len(points)} pts to {out_ply}", flush=True)

print("Done")
