from typing import Dict

import torch
from ema_pytorch import EMA
from loguru import logger
from torch import optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

from models.flow_bridge import OTFlowBridge, LatentOTFlowBridge
from models.p2pb import P2PB
from models.unet_pvc import PVCNN2Unet


def load_optim_sched(cfg: Dict, model: torch.nn.Module, model_ckpt: str = None) -> tuple:
    """Load optimizer and scheduler according to the configuration."""

    lr = cfg.training.optimizer.lr
    wd = cfg.training.optimizer.weight_decay
    betas = (cfg.training.optimizer.beta1, cfg.training.optimizer.beta2)

    # Check if model has freq_transformer (latent mode) for per-param-group optimizer
    has_freq_transformer = hasattr(model, 'freq_transformer')

    if has_freq_transformer:
        ft_lr_mult = float(cfg.training.optimizer.get("freq_transformer_lr_mult", 1.0))
        wa_lr_mult = float(cfg.training.optimizer.get("write_attn_lr_mult", ft_lr_mult))
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters()
                           if "freq_transformer" not in n and "write_attn" not in n
                           and "cond_" not in n and p.requires_grad],
                "weight_decay": wd,
                "lr": lr,
                "name": "backbone",
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if "freq_transformer" in n and p.requires_grad],
                "weight_decay": 0.0,
                "lr": lr * ft_lr_mult,
                "name": "freq_transformer",
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if ("write_attn" in n or "cond_" in n) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": lr * wa_lr_mult,
                "name": "write_attn",
            },
        ]
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
    else:
        param_groups = model.parameters()

    # setup optimizer
    if cfg.training.optimizer.type == "Adam":
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=wd,
            betas=betas,
        )
    elif cfg.training.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
        )
    else:
        raise NotImplementedError(cfg.training.optimizer.type)

    # setup lr scheduler
    if cfg.training.scheduler.type == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.training.scheduler.lr_gamma)
    elif cfg.training.scheduler.type == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.9)
    elif cfg.training.scheduler.type == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.scheduler.get("T_max", cfg.training.steps),
            eta_min=cfg.training.scheduler.get("eta_min", 1e-6),
        )
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if model_ckpt is not None and not cfg.restart:
        try:
            optimizer.load_state_dict(model_ckpt["optimizer_state"])
        except Exception as e:
            logger.warning(e)
        try:
            lr_scheduler.load_state_dict(model_ckpt["scheduler_state"])
            resumed_step = model_ckpt.get("step", "unknown")
            logger.info(f"Scheduler state restored from step {resumed_step}")
        except (KeyError, Exception) as e:
            logger.warning(f"Scheduler state not restored (will restart from step 0): {e}")
    logger.info("Optimizer and scheduler prepared")

    return optimizer, lr_scheduler


def load_model(cfg: Dict) -> torch.nn.Module:
    """
    Load a model based on the given configuration.

    Args:
        cfg (Dict): The configuration dictionary.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = PVCNN2Unet(cfg)
    logger.info(
        f"Generated model with following number of params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}"
    )
    return model


def load_diffusion(cfg: Dict) -> tuple:
    """Loads diffusion model including backbone and prepares it for training."""
    # setup model
    backbone = load_model(cfg).to(cfg.local_rank)

    bridge_type = cfg.diffusion.get("bridge_type", "dsb")
    model_type = str(getattr(cfg.model, "model_type", "pvd")).lower()

    if bridge_type == "otcfm" and model_type in ("latent_otcfm", "latentp2pb", "latent"):
        model = LatentOTFlowBridge(cfg=cfg, model=backbone)
    elif bridge_type == "otcfm":
        model = OTFlowBridge(cfg=cfg, model=backbone)
    else:
        model = P2PB(cfg=cfg, model=backbone)

    gpu = cfg.local_rank

    model = model.cuda()

    # setup DDP model
    if cfg.distribution_type == "multi":

        def ddp_transform(m):
            return DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)

        model.multi_gpu_wrapper(ddp_transform)

    # setup data parallel model
    elif cfg.distribution_type == "single":

        def dp_transform(m):
            return DataParallel(m)

        model.multi_gpu_wrapper(dp_transform)

    # load the model weights
    cfg.start_step = 0
    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path, map_location=torch.device("cpu"), weights_only=False)

        if not cfg.restart:
            cfg.start_step = ckpt["step"] + 1

            try:
                model_state = ckpt["model_state"]

                if cfg.distribution_type in ["multi", "single"]:
                    model_dict = extract_from_state_dict(model_state, "model.")
                    ema_dict = extract_from_state_dict(model_state, "ema.")
                else:
                    model_dict = extract_from_state_dict(model_state, "model.module.")
                    ema_dict = extract_from_state_dict(model_state, "ema.")

                model.model.load_state_dict(model_dict, strict=False)
                if cfg.use_ema:
                    if ema_dict != {}:
                        try:
                            model.ema.load_state_dict(ema_dict)
                            logger.success("Loaded EMA from checkpoint!")
                        except RuntimeError:
                            logger.warning("EMA shape mismatch — reinitializing EMA from model.")
                            model.ema = EMA(model.model, beta=0.999)

                # Load FreqTransformer weights if present
                if hasattr(model, "freq_transformer"):
                    ft_dict = extract_from_state_dict(model_state, "freq_transformer.")
                    if ft_dict:
                        model.freq_transformer.load_state_dict(ft_dict, strict=False)
                        logger.success("Loaded FreqTransformer from checkpoint!")
                    else:
                        logger.warning("No freq_transformer keys in checkpoint — starting fresh (new latent layers).")

                logger.success("Loaded Model from checkpoint!")

            except RuntimeError as e:
                logger.warning("Could not load model state dict. Trying to load without strict flag.")
                logger.warning(e)
                model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            logger.warning("Restarting training from existing checkpoint backbone only.")
            logger.warning("Loading backbone weights; optimizer + latent layers start fresh.")
            model_state = ckpt["model_state"]
            model_dict = extract_from_state_dict(model_state, "model.")
            try:
                # only load the model parameters and let rest start from scratch
                model.model.load_state_dict(model_dict, strict=False)
                # Load FreqTransformer weights even on restart
                if hasattr(model, "freq_transformer"):
                    ft_dict = extract_from_state_dict(model_state, "freq_transformer.")
                    if ft_dict:
                        model.freq_transformer.load_state_dict(ft_dict, strict=False)
                        logger.success("Loaded FreqTransformer from checkpoint!")
                # set ema
                if cfg.use_ema:
                    model.ema = EMA(model.model, beta=0.999)
            except RuntimeError:
                logger.warning("Could not load model state dict. Trying to load adaptively.")
                load_matched_weights(model.model, model_dict)
                if hasattr(model, "freq_transformer"):
                    ft_dict = extract_from_state_dict(model_state, "freq_transformer.")
                    if ft_dict:
                        model.freq_transformer.load_state_dict(ft_dict, strict=False)
                if cfg.use_ema:
                    model.ema = EMA(model.model, beta=0.999)
        logger.warning("Loaded model from %s" % cfg.model_path)
    else:
        ckpt = None

    torch.cuda.empty_cache()
    return model, ckpt


def extract_from_state_dict(state_dict: Dict, pattern: str) -> Dict:
    """
    Extracts key-value pairs from a state dictionary based on a given pattern.

    Args:
        state_dict (Dict): The state dictionary to extract from.
        pattern (str): The pattern to match the keys against.

    Returns:
        Dict: A dictionary containing the matched key-value pairs.
    """
    matched_kv_pairs = {k.replace(pattern, ""): v for k, v in state_dict.items() if k.startswith(pattern)}
    return matched_kv_pairs


def load_matched_weights(model: torch.nn.Module, state_dict_to_load: Dict):
    """
    Loads matched weights from a state dictionary into a model.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        state_dict_to_load (Dict): The state dictionary containing the weights to load.

    Returns:
        None
    """
    own_state = model.state_dict()
    for name, param in state_dict_to_load.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
            except Exception as e:
                print(f"Failed to load parameter {name}. Exception: {e}")
        elif "." in name:
            sub_module_names = name.split(".")
            sub_module = model
            for sub_module_name in sub_module_names[:-1]:
                if hasattr(sub_module, sub_module_name):
                    sub_module = getattr(sub_module, sub_module_name)
                else:
                    break
            else:
                sub_param_name = sub_module_names[-1]
                if hasattr(sub_module, sub_param_name):
                    sub_param = getattr(sub_module, sub_param_name)
                    if sub_param.shape == param.shape:
                        sub_param.data.copy_(param)
        else:
            print(f"Parameter {name} not found in model. Skipping.")
