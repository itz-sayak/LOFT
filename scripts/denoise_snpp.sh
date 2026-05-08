#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/denoise_snpp.sh <data_root> [model_path] [gpu_id]"
    exit 1
fi

DATA_ROOT="$1"
MODEL_PATH="${2:-/mnt/zone/B/NEW/P2P-Bridge-OT-real-latent/checkpoints/PVDL_SNPP_latent/step_230000.pth}"
GPU_ID="${3:-1}"
SCENE_LIST="${4:-splits/snpp_test_valid.txt}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing scene list: $SCENE_LIST"
    exit 1
fi

for scene in $(cat "$SCENE_LIST"); do
    echo "Processing $scene"
    BASE="$DATA_ROOT/$scene"
    OUT="$BASE/predictions_dino/P2SB/PVDL-SNPP-latent_iphone-dino_230000_10_ema.ply"

    mkdir -p "$(dirname "$OUT")"
    python -u denoise_room.py \
        --room_path "$BASE/scans/iphone_dino.ply" \
        --model_path "$MODEL_PATH" \
        --steps 10 \
        --k 4 \
        --out_path "$OUT"
done

echo "Running metrics evaluation"
python -u evaluate_rooms.py --data_root "$DATA_ROOT" --dataset snpp --suffix _dino