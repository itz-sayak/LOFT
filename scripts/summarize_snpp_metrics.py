#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List

import pandas as pd


def pick_row(df: pd.DataFrame, pattern: str) -> pd.Series:
    if "model_config" not in df.columns or df.empty:
        raise ValueError("Missing model_config column or empty metrics file")

    if pattern:
        matched = df[df["model_config"].astype(str).str.contains(pattern, regex=True, na=False)]
        if not matched.empty:
            return matched.iloc[-1]

    return df.iloc[-1]


def load_scene_row(scene_root: str, model_dir: str, suffix: str, pattern: str) -> Dict[str, float]:
    metrics_file = os.path.join(scene_root, "metrics", model_dir, f"metrics{suffix}.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(metrics_file)

    df = pd.read_csv(metrics_file)
    row = pick_row(df, pattern)

    return {
        "point_dist": float(row["point_dist"]),
        "face_dist": float(row["face_dist"]),
        "cd_pred_gt": float(row["cd_pred_gt"]),
        "cd_gt_pred": float(row["cd_gt_pred"]),
    }


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SNPP metrics for two models over a scene list")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene_file", type=str, default="splits/snpp_test_valid.txt")
    parser.add_argument("--suffix", type=str, default="_dino")

    parser.add_argument("--p2p_model_dir", type=str, default="P2SB")
    parser.add_argument("--p2p_label", type=str, default="P2P orig")
    parser.add_argument("--p2p_pattern", type=str, default="")

    parser.add_argument("--latent_model_dir", type=str, default="P2SB")
    parser.add_argument("--latent_label", type=str, default="Latent")
    parser.add_argument("--latent_pattern", type=str, default="250000_10_ema")
    args = parser.parse_args()

    with open(args.scene_file, "r", encoding="utf-8") as f:
        scenes = [line.strip() for line in f if line.strip()]

    rows: List[Dict[str, str]] = []
    agg = {
        args.p2p_label: {"point_dist": 0.0, "face_dist": 0.0, "cd_pred_gt": 0.0, "cd_gt_pred": 0.0},
        args.latent_label: {"point_dist": 0.0, "face_dist": 0.0, "cd_pred_gt": 0.0, "cd_gt_pred": 0.0},
    }

    for scene in scenes:
        scene_root = os.path.join(args.data_root, scene)

        p2p = load_scene_row(scene_root, args.p2p_model_dir, args.suffix, args.p2p_pattern)
        lat = load_scene_row(scene_root, args.latent_model_dir, args.suffix, args.latent_pattern)

        rows.append(
            {
                "Scene": scene,
                "Model": args.p2p_label,
                "point_dist": fmt(p2p["point_dist"]),
                "face_dist": fmt(p2p["face_dist"]),
                "cd_pred_gt": fmt(p2p["cd_pred_gt"]),
                "cd_gt_pred": fmt(p2p["cd_gt_pred"]),
            }
        )
        rows.append(
            {
                "Scene": scene,
                "Model": args.latent_label,
                "point_dist": fmt(lat["point_dist"]),
                "face_dist": fmt(lat["face_dist"]),
                "cd_pred_gt": fmt(lat["cd_pred_gt"]),
                "cd_gt_pred": fmt(lat["cd_gt_pred"]),
            }
        )

        for k in agg[args.p2p_label]:
            agg[args.p2p_label][k] += p2p[k]
            agg[args.latent_label][k] += lat[k]

    n = float(len(scenes))
    rows.append(
        {
            "Scene": "Mean",
            "Model": args.p2p_label,
            "point_dist": fmt(agg[args.p2p_label]["point_dist"] / n),
            "face_dist": fmt(agg[args.p2p_label]["face_dist"] / n),
            "cd_pred_gt": fmt(agg[args.p2p_label]["cd_pred_gt"] / n),
            "cd_gt_pred": fmt(agg[args.p2p_label]["cd_gt_pred"] / n),
        }
    )
    rows.append(
        {
            "Scene": "Mean",
            "Model": args.latent_label,
            "point_dist": fmt(agg[args.latent_label]["point_dist"] / n),
            "face_dist": fmt(agg[args.latent_label]["face_dist"] / n),
            "cd_pred_gt": fmt(agg[args.latent_label]["cd_pred_gt"] / n),
            "cd_gt_pred": fmt(agg[args.latent_label]["cd_gt_pred"] / n),
        }
    )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
