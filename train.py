import os
import sys
import time

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from dataloaders.dataloader import get_dataloader, save_iter
from dataloaders.punet import get_alignment_clean
from metrics.emd_assignment import emd_module
from models.evaluation import evaluate
from models.model_loader import load_diffusion, load_optim_sched
from models.train_utils import get_data_batch, getGradNorm, set_seed, setup_output_subdirs, to_cuda
from utils.args import parse_args


def _print_eval_row(step: int, metrics: dict) -> None:
    print()
    print(f"  {'Step':>8}  {'CD':>10}  {'EMD':>10}  {'MSE':>10}  {'CD_noisy':>10}  {'EMD_noisy':>10}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    print(
        f"  {step:>8d}  {metrics.get('cd', float('nan')):>10.6f}  {metrics.get('emd', float('nan')):>10.6f}  "
        f"{metrics.get('mse', float('nan')):>10.6f}  {metrics.get('cd_noisy', float('nan')):>10.6f}  "
        f"{metrics.get('emd_noisy', float('nan')):>10.6f}"
    )
    print()


def init_processes(rank: int | str, size: int, fn: callable, args: DictConfig) -> None:
    """Initialize the distributed environment.

    Args:
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        fn (function): Function to run.
        args (DictConfig): Configuration.
    """
    torch.cuda.set_device(rank)
    args.local_rank = rank
    args.global_rank = rank
    args.global_size = size
    args.gpu = rank

    # usual env init
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)

    fn(args)

    dist.barrier()
    dist.destroy_process_group()


def train(cfg: DictConfig) -> None:
    is_main_process = cfg.local_rank == 0

    logger.remove()

    if is_main_process:
        (outf_syn,) = setup_output_subdirs(cfg.output_dir, "output")
        cfg.outf_syn = outf_syn
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            + "<level>{level: <8}</level> | "
            + "<level>{message}</level>"
        )
        logger.add(sys.stdout, level="WARNING", format=fmt)

    set_seed(cfg)
    torch.cuda.empty_cache()

    train_loader, val_loader, train_sampler, val_sampler = get_dataloader(cfg)

    model, ckpt = load_diffusion(cfg)

    optimizer, lr_scheduler = load_optim_sched(cfg, model, ckpt)
    logger.info("Training with config {}", cfg.config)

    # setup alignment function for PUNet
    if cfg.data.dataset == "PUNet":
        aligner = emd_module.emdModule()
        emd_align = get_alignment_clean(aligner)

        @torch.no_grad()
        def align_fn(noisy, clean):
            align_idxs = emd_align(noisy, clean).detach().long()
            align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
            clean = torch.gather(clean, -1, align_idxs)
            # use the indices to align the clean points
            return clean

    else:
        align_fn = None

    if is_main_process:
        wandb.login()
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.wandb_entity,
        )
        try:
            wandb.watch(model, log="all", log_freq=cfg.training.log_interval * 10)
        except Exception as e:
            logger.warning("Could not watch model. Skipping.")
            logger.warning(e)

    ampscaler = torch.amp.GradScaler('cuda', enabled=cfg.training.amp)

    train_iter = save_iter(train_loader, train_sampler)
    torch.cuda.empty_cache()
    total_steps = cfg.training.steps
    start_step = cfg.start_step
    pbar = (
        tqdm(
            range(start_step, total_steps),
            total=total_steps - start_step,
            initial=0,
            unit="step",
            dynamic_ncols=True,
            bar_format=(
                "{percentage:3.0f}%|{bar:42}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )
        if is_main_process
        else range(start_step, total_steps)
    )
    t_start = time.time()

    # One-time startup diagnostics
    if is_main_process:
        logger.warning(
            "[cfg] ot_update_freq={} (OT coupling runs every {} step(s)). "
            "Per-component timing will print for the first 5 training steps.",
            model.ot_update_freq, model.ot_update_freq,
        )

    for step in pbar:
        _do_step_timing = is_main_process and model._timing_countdown > 0
        if _do_step_timing:
            torch.cuda.synchronize()
            _t_step0 = time.perf_counter()

        optimizer.zero_grad()

        # update the sampler for multi-node training
        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(train_loader))

        loss_accum = torch.tensor(0.0, dtype=torch.float32, device=cfg.local_rank)

        for accum_iter in range(cfg.training.accumulation_steps):
            # ── data loading ──────────────────────────────────────────────
            if _do_step_timing:
                torch.cuda.synchronize()
                _t_data0 = time.perf_counter()
            next_batch = next(train_iter)
            next_batch = to_cuda(next_batch, cfg.local_rank)
            data = next_batch
            data_batch = get_data_batch(batch=data, cfg=cfg, align_fn=align_fn)
            x_gt = data_batch["x_gt"]
            x_cond = data_batch["x_cond"]
            x_start = data_batch["x_start"]
            if _do_step_timing:
                torch.cuda.synchronize()
                _data_ms = (time.perf_counter() - _t_data0) * 1000.0

            # ── OT-CFM forward (OT+forward timing printed inside model) ───
            if _do_step_timing:
                torch.cuda.synchronize()
                _t_fwd0 = time.perf_counter()
            # OT-CFM: x0=noisy (x_start), x1=clean (x_gt).
            # For ScanNet++: concatenate DINO features as extra channels into x_start
            # so the bridge receives [B, 3+384, N] and extracts xyz+DINO internally.
            x_start_bridge = (
                torch.cat([x_start, x_cond], dim=1) if x_cond is not None else x_start
            )
            loss = model.train_loss(x_start_bridge, x_gt)
            loss /= cfg.training.accumulation_steps
            loss_accum += loss.detach()
            if _do_step_timing:
                torch.cuda.synchronize()
                _train_loss_ms = (time.perf_counter() - _t_fwd0) * 1000.0

            # ── backward ──────────────────────────────────────────────────
            if _do_step_timing:
                torch.cuda.synchronize()
                _t_bwd0 = time.perf_counter()
            ampscaler.scale(loss).backward()
            if _do_step_timing:
                torch.cuda.synchronize()
                _bwd_ms = (time.perf_counter() - _t_bwd0) * 1000.0

        # ── optimizer step ────────────────────────────────────────────────
        if _do_step_timing:
            torch.cuda.synchronize()
            _t_opt0 = time.perf_counter()
        ampscaler.unscale_(optimizer)
        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        scale_before = ampscaler.get_scale()
        ampscaler.step(optimizer)
        ampscaler.update()
        # Only advance scheduler (and EMA) when the optimizer actually stepped.
        # If AMP detected inf/NaN, get_scale() drops — that signals a skipped step.
        if ampscaler.get_scale() >= scale_before:
            lr_scheduler.step()

        if model.ema is not None:
            model.ema.update()
        if _do_step_timing:
            torch.cuda.synchronize()
            _opt_ms = (time.perf_counter() - _t_opt0) * 1000.0
            _total_ms = (time.perf_counter() - _t_step0) * 1000.0
            _misc_ms = _total_ms - _data_ms - _train_loss_ms - _bwd_ms - _opt_ms
            logger.warning(
                "[timing step {:>4d}] data={:.1f}ms  train_loss={:.1f}ms"
                "  (ot={:.1f}ms  fwd={:.1f}ms)"
                "  backward={:.1f}ms  optim={:.1f}ms  misc={:.1f}ms  total={:.1f}ms",
                step,
                _data_ms, _train_loss_ms,
                model._last_ot_ms, _train_loss_ms - model._last_ot_ms,
                _bwd_ms, _opt_ms, _misc_ms, _total_ms,
            )

        if cfg.distribution_type == "multi":
            dist.all_reduce(loss_accum)

        if step % cfg.training.log_interval == 0 and is_main_process:
            loss_accum /= cfg.global_size
            loss_accum = loss_accum.item()
            netpNorm, netgradNorm = getGradNorm(model.model)
            lr_now = optimizer.param_groups[0]["lr"]

            try:
                gpu_mem_gb = torch.cuda.memory_reserved(cfg.local_rank) / 1024**3
                gpu_mem_str = f"{gpu_mem_gb:.1f}G"
            except Exception:
                gpu_mem_str = "n/a"

            pbar.set_postfix(
                loss=f"{loss_accum:.4f}",
                pNorm=f"{netpNorm:.1f}",
                gNorm=f"{netgradNorm:.4f}",
                lr=f"{lr_now:.2e}",
                mem=gpu_mem_str,
            )

            tqdm.write(
                f"  step {step+1:>7d}/{total_steps}  "
                f"loss={loss_accum:.6f}  "
                f"pNorm={netpNorm:.2f}  "
                f"gNorm={netgradNorm:.6f}  "
                f"lr={lr_now:.2e}  "
                f"mem={gpu_mem_str}"
            )

            wandb.log(
                {
                    "loss": loss_accum,
                    "netpNorm": netpNorm,
                    "netgradNorm": netgradNorm,
                    "lr": lr_now,
                },
                step=step,
            )

        if (step + 1) % cfg.training.save_interval == 0:
            if is_main_process:
                save_dict = {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict(),
                }
                ckpt_path = "%s/step_%d.pth" % (cfg.output_dir, step + 1)
                torch.save(save_dict, ckpt_path)
                tqdm.write(f"  [step {step+1:>7d}] Checkpoint saved -> {ckpt_path}")

            if cfg.distribution_type == "multi":
                dist.barrier()
                map_location = {"cuda:%d" % 0: "cuda:%d" % cfg.local_rank}
                model.load_state_dict(
                    torch.load(
                        "%s/step_%d.pth" % (cfg.output_dir, step + 1),
                        map_location=map_location,
                        weights_only=False,
                    )["model_state"]
                )

        if (step + 1) % cfg.training.viz_interval == 0:
            if cfg.distribution_type == "multi":
                dist.barrier()

            model.eval()
            if is_main_process:
                tqdm.write(
                    f"\n  {'-' * 68}\n"
                    f"  Eval @ step {step+1:>7d} / {total_steps}\n"
                    f"  {'-' * 68}"
                )
                try:
                    metrics = evaluate(model, val_loader, cfg, step + 1)
                    if metrics:
                        _print_eval_row(step + 1, metrics)
                except Exception as e:
                    tqdm.write(f"  [WARN] Eval failed at step {step+1}: {e}")

            torch.cuda.empty_cache()
            model.train()

    if is_main_process:
        elapsed_min = (time.time() - t_start) / 60.0
        tqdm.write(f"\n  Training complete ({elapsed_min:.1f} min)")

    wandb.finish()


if __name__ == "__main__":
    opt = parse_args()

    # save the opt to output_dir
    save_data = DictConfig({})
    save_data.data = opt.data
    save_data.diffusion = opt.diffusion
    save_data.model = opt.model
    save_data.sampling = opt.sampling
    save_data.training = opt.training
    OmegaConf.save(save_data, os.path.join(opt.output_dir, "opt.yaml"))

    opt.ngpus_per_node = torch.cuda.device_count()

    torch.set_float32_matmul_precision("high")

    if opt.distribution_type == "multi":
        # setup configurations
        opt.world_size = opt.ngpus_per_node * opt.world_size
        opt.training.bs = int(opt.training.bs / opt.ngpus_per_node)
        opt.sampling.bs = opt.training.bs

        mp.spawn(init_processes, nprocs=opt.world_size, args=(opt.world_size, train, opt))
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        opt.gpu = 0
        train(opt)
