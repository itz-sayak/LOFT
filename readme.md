<p align="center">
  <h1 align="center">Semantics Know the Shape: Latent-Conditioned Flow Matching for 3D Point Cloud Denoising</h1>
  <p align="center">
    <strong>LOFT — Latent-Guided Optimal Flow Transport</strong>
  </p>
</p>

---

## Overview

LOFT is a latent-conditioned flow matching framework for 3D point cloud denoising. It combines Optimal Transport Conditional Flow Matching (OT-CFM) with a semantic latent conditioning pipeline: a frozen SemanticAutoencoder extracts geometric latent tokens from noisy input, a trainable FreqEncodingTransformer refines them at each timestep, and these tokens are injected into the PVCNN2Unet denoising backbone via cross-attention at five architectural levels. The result is a velocity field that transports noisy point patches to clean geometry via a straight-line OT-optimal trajectory.

**Paper:** *Semantics Know the Shape: Latent-Conditioned Flow Matching for 3D Point Cloud Denoising*

---

## Requirements

Tested with Python 3.10, PyTorch 2.5.1+cu121, CUDA 12.1, Ubuntu 22.04, RTX 4090 24 GB.

Create a conda environment:

```bash
conda create -n loft python=3.10
conda activate loft
```

Install PyTorch:

```bash
conda install pytorch==2.5.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia --yes
```

Install all other dependencies and compile custom CUDA extensions:

```bash
sh install.sh
```

---

## Data Preparation

Download the PUNet dataset meshes and place them under `data/`:

```
data/
  10000_poisson/
  30000_poisson/
  50000_poisson/
```

The data directory path is configured in `configs/PVDS_PUNet_latent.yaml` under `data.data_dir`.

---

## Training

```bash
python train.py --config configs/PVDS_PUNet_latent.yaml
```

Checkpoints are saved to `checkpoints/otcfm_latent/` every 25,000 steps. Training runs for 450,000 steps at batch size 32 (~1.1 s/step on RTX 4090, ~10.1 GB VRAM).

Key config options in `configs/PVDS_PUNet_latent.yaml`:

| Option | Default | Description |
|--------|---------|-------------|
| `model.ae_ckpt` | path to `ae_epoch_675.pth` | Frozen SemanticAE checkpoint |
| `diffusion.latent_loss_weight` | 0.3 | Weight for cosine latent consistency loss |
| `training.bs` | 32 | Batch size |
| `diffusion.sampling_timesteps` | 10 | Euler ODE steps at inference |

---

## Evaluation

```bash
python denoise_object.py --config configs/PVDS_PUNet_latent.yaml --model_path checkpoints/otcfm_latent/ckpt_best.pth
python evaluate_objects.py
```

---

## Architecture

See [architecture.md](architecture.md) for a full description of every component and its data flow, suitable for building block diagrams.

**Parameter summary:**

| Component | Parameters | Status |
|-----------|----------:|--------|
| SemanticAutoencoder | 5,985,653 | Frozen |
| PVCNN2Unet backbone | ~19.4M | Trainable (lr = 3×10⁻⁴) |
| LatentWriteAttention (×5) | ~1.07M | Trainable (lr = 9×10⁻⁴) |
| FreqEncodingTransformer | 17,389,184 | Trainable (lr = 9×10⁻⁴) |
| **Total trainable** | **~37.9M** | |

---

## Repository Structure

```
configs/          Training configs (use PVDS_PUNet_latent.yaml for LOFT)
models/
  autoencoder.py          SemanticAutoencoder (frozen AE)
  freq_encoding_transformer.py  FreqEncodingTransformer
  flow_bridge.py          OTFlowBridge + LatentOTFlowBridge
  unet_pvc.py             PVCNN2Unet + LatentWriteAttention
  model_loader.py         Model instantiation + per-param-group optimizer
dataloaders/      Dataset loaders
metrics/          Chamfer Distance, EMD evaluation
third_party/      PVCNN, OpenPoints libraries
architecture.md   Full architecture description for block diagram
```

---

## Acknowledgements

This work uses the PVCNN architecture. The SemanticAutoencoder and FreqEncodingTransformer are original contributions of this project.

## Recent changes (2026-05-03)

- `latent_film` explicitly disabled for ScanNet++ in `configs/PVDL_SNPP_latent.yaml` (`latent_film: false`).
  - Reason: avoids train/inference mismatch when using AE FiLM with noisy real-scene inputs.
  - PUNet latent config continues to use `latent_film: true` for upsampling tasks.

## Applying the change (restart note)

If a ScanNet++ training process is already running it will continue using the config that was loaded at start. To apply the `latent_film: false` change:

```bash
# attach to the otcfm screen and stop the process safely
screen -r otcfm
# inside screen: Ctrl-C to stop the job, or exit the training loop cleanly

# then restart with the updated config
conda activate deepfill
cd /mnt/zone/B/NEW/P2P-Bridge-OT-real-latent
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u train.py --config configs/PVDL_SNPP_latent.yaml 2>&1 | tee logs/train_snpp_latent.log
```

Logs are written to `logs/` and checkpoints to `checkpoints/` as configured.
