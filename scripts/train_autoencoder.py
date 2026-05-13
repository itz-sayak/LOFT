import sys, os
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from models.autoencoder import GeomTokenAE
from dataloaders.punet import PUNetDatasetWrapper as PUNetDataset
from dataloaders.scannetpp import ScanNetPP as ScanNetPPDataset

# ---------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------
def chamfer_distance(x, y):
    # x,y: [B,N,3]
    diff_x = torch.cdist(x, y, p=2)  # [B,N,N]
    mins1 = diff_x.min(dim=2)[0]
    mins2 = diff_x.min(dim=1)[0]
    loss = (mins1.mean(dim=1) + mins2.mean(dim=1)).mean()
    return loss


def reconstruction_loss(pred, target, l1_weight):
    loss_cd = chamfer_distance(pred, target)
    loss_l1 = torch.mean(torch.abs(pred - target))
    return loss_cd + l1_weight * loss_l1, loss_cd, loss_l1


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_autoencoder(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    dataset_type = getattr(cfg.data, "dataset", "PUNet").lower()
    if dataset_type == "scannetpp":
        train_ds = ScanNetPPDataset(
            root=cfg.data.data_dir,
            mode="training",
            augment=cfg.data.augment,
            train_split_path=getattr(cfg.data, "train_split_path", None),
            val_split_path=getattr(cfg.data, "val_split_path", None),
        )
    else:
        train_ds = PUNetDataset(
            root=cfg.data.data_dir,
            split="train",
            npoints=cfg.data.npoints,
            augment=cfg.data.augment,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.bs,
        shuffle=True,
        num_workers=cfg.data.workers,
        drop_last=True,
    )

    # ---- Model ----
    model = GeomTokenAE(
        in_dim=3,
        feat_dim=512,
        use_dino=getattr(cfg.model, "use_dino", False), 
        dino_dim=getattr(cfg.model, "dino_dim", 384), 
        sem_dim=getattr(cfg.model, "sem_dim", 128),
        use_cross_attn=getattr(cfg.model, "use_cross_attn", True),
        use_offset_attn=getattr(cfg.model, "use_offset_attn", True),
    ).to(device)

    resume_from = getattr(cfg.training, "resume_from", None)
    start_epoch = int(getattr(cfg.training, "start_epoch", 0) or 0)
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        if start_epoch <= 0:
            match = re.search(r"ae_epoch_(\d+)\.pth$", os.path.basename(resume_from))
            if match:
                start_epoch = int(match.group(1))
        logger.info("Resuming AE from {} at epoch {}", resume_from, start_epoch)

    train_cfg = cfg.training
    opt_cfg = getattr(train_cfg, "optimizer", None)
    sched_cfg = getattr(train_cfg, "scheduler", None)

    lr = getattr(opt_cfg, "lr", 1e-4)
    beta1 = getattr(opt_cfg, "beta1", 0.9)
    beta2 = getattr(opt_cfg, "beta2", 0.999)
    weight_decay = getattr(opt_cfg, "weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    if getattr(sched_cfg, "type", "constant").lower() == "cosine":
        min_lr = getattr(sched_cfg, "min_lr", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.epochs,
            eta_min=min_lr,
        )
    else:
        scheduler = None

    l1_weight = getattr(train_cfg, "l1_weight", 0.1)
    denoise_weight = getattr(train_cfg, "denoise_weight", 1.0)
    clean_weight = getattr(train_cfg, "clean_weight", 1.0)
    latent_consistency_weight = getattr(train_cfg, "latent_consistency_weight", 0.05)
    latent_l2_weight = getattr(train_cfg, "latent_l2_weight", 1.0e-4)
    grad_clip = getattr(getattr(train_cfg, "grad_clip", None), "value", None)

    logger.info(
        "Autoencoder params: {:.2f} M, LR={}, clean_weight={}, denoise_weight={}, latent_consistency_weight={}",
        sum(p.numel() for p in model.parameters()) / 1e6,
        lr,
        clean_weight,
        denoise_weight,
        latent_consistency_weight,
    )

    # ---- Training ----
    total_epochs = int(cfg.training.epochs)
    end_epoch = start_epoch + total_epochs
    save_best_only = bool(getattr(cfg.training, "save_best_only", False))
    best_metric_name = str(getattr(cfg.training, "best_metric", "loss")).lower()
    if best_metric_name not in ["loss", "clean_cd", "noisy_cd", "latent_consistency"]:
        raise ValueError(f"Unsupported best_metric '{best_metric_name}'")
    best_metric_value = float("inf")
    best_ckpt_path = None

    def get_metric_value(name, loss_v, clean_cd_v, noisy_cd_v, consistency_v):
        if name == "loss":
            return loss_v
        if name == "clean_cd":
            return clean_cd_v
        if name == "noisy_cd":
            return noisy_cd_v
        return consistency_v

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        epoch_clean_cd = 0.0
        epoch_noisy_cd = 0.0
        epoch_consistency = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            noisy_xyz = batch.get("noisy_points", batch.get("points")).to(device)
            clean_xyz = batch.get("clean_points", noisy_xyz).to(device)

            optimizer.zero_grad()

            recon_clean, latent_clean = model(clean_xyz)
            recon_noisy, latent_noisy = model(noisy_xyz)

            clean_recon_loss, clean_cd, _ = reconstruction_loss(recon_clean, clean_xyz, l1_weight)
            noisy_recon_loss, noisy_cd, _ = reconstruction_loss(recon_noisy, clean_xyz, l1_weight)
            latent_consistency = torch.mean((latent_noisy - latent_clean.detach()) ** 2)
            latent_l2 = 0.5 * (latent_clean.pow(2).mean() + latent_noisy.pow(2).mean())

            loss = (
                clean_weight * clean_recon_loss
                + denoise_weight * noisy_recon_loss
                + latent_consistency_weight * latent_consistency
                + latent_l2_weight * latent_l2
            )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += loss.item()
            epoch_clean_cd += clean_cd.item()
            epoch_noisy_cd += noisy_cd.item()
            epoch_consistency += latent_consistency.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_clean_cd = epoch_clean_cd / len(train_loader)
        avg_noisy_cd = epoch_noisy_cd / len(train_loader)
        avg_consistency = epoch_consistency / len(train_loader)

        logger.info(
            "Epoch {}: loss={:.6f}, clean_cd={:.6f}, noisy_cd={:.6f}, latent_consistency={:.6f}, lr={:.2e}",
            epoch,
            avg_loss,
            avg_clean_cd,
            avg_noisy_cd,
            avg_consistency,
            optimizer.param_groups[0]["lr"],
        )

        # ---- Save checkpoint ----
        os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
        if save_best_only:
            current_metric = get_metric_value(
                best_metric_name,
                avg_loss,
                avg_clean_cd,
                avg_noisy_cd,
                avg_consistency,
            )
            if current_metric < best_metric_value:
                best_metric_value = current_metric
                path = os.path.join(cfg.training.ckpt_dir, f"ae_best_{best_metric_name}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "best_metric": best_metric_name,
                        "best_metric_value": best_metric_value,
                        "model_state_dict": model.state_dict(),
                    },
                    path,
                )
                best_ckpt_path = path
                logger.success(
                    "Saved new best checkpoint: {} ({}={:.6f})",
                    path,
                    best_metric_name,
                    best_metric_value,
                )
        else:
            if (epoch % cfg.training.save_interval == 0) or (epoch == end_epoch):
                path = os.path.join(cfg.training.ckpt_dir, f"ae_epoch_{epoch}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                }, path)
                logger.success(f"Saved checkpoint: {path}")

    if save_best_only and best_ckpt_path is not None:
        logger.info(
            "Pretraining complete. Best checkpoint: {} ({}={:.6f})",
            best_ckpt_path,
            best_metric_name,
            best_metric_value,
        )
    else:
        logger.info("Pretraining complete.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    from types import SimpleNamespace
    def to_ns(d): 
        return SimpleNamespace(**{k: to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    cfg = to_ns(cfg)

    if not hasattr(cfg.training, "epochs"): cfg.training.epochs = 150
    if not hasattr(cfg.training, "save_interval"): cfg.training.save_interval = 10
    if not hasattr(cfg.training, "ckpt_dir"): cfg.training.ckpt_dir = "checkpoints_ae"

    train_autoencoder(cfg)


