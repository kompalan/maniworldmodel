import os
import random
import time
from dataclasses import asdict
from functools import partial
from typing import Any

import numpy as np
import stable_pretraining as spt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from tqdm import tqdm
import dotenv

from .config import *
from .dataset import ManiSkillTrajectoryDataset
from .jepa import JEPA
from .module import MLP, ARPredictor, Embedder, SIGReg
from .utils import Logger, autocast, to_device, get_model

def lejepa_forward(batch, model: JEPA, sigreg: SIGReg, args: Args):
    """encode observations, predict next states, compute losses."""

    ctx_len = args.model_args.history_size
    n_preds = args.model_args.predictor.num_preds
    lambd = args.loss_args.sigreg_weight

    output = model.encode(batch)

    emb = output["emb"]  # (B, T, D), T = ctx_len + n_preds
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss (fp32: SIGReg's cos/sin statistic is too coarse in bf16)
    output["pred_loss"] = (pred_emb.float() - tgt_emb.float()).pow(2).mean()
    with torch.autocast(device_type=emb.device.type, enabled=False):
        output["sigreg_loss"] = sigreg(emb.float().transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    return output

def get_optimizer(args: Args, model: nn.Module):
    opt = args.training_args.optimizer_args
    if opt.type == OptimizerType.ADAMW:
        return torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    if opt.type == OptimizerType.SGD:
        return torch.optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer {opt.type}")


def get_scheduler(args: Args, optimizer, total_steps: int):
    if args.training_args.lr_scheduler_type == LRSchedulerType.LINEAR_WARMUP_COSINE_ANNEALING:
        warmup = min(args.training_args.warmup_steps, total_steps // 10)
        return spt.optim.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup, total_steps)
    raise ValueError(f"unknown scheduler {args.training_args.lr_scheduler_type}")


def do_epoch(
        args: Args, 
        model: JEPA, 
        sigreg: SIGReg, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any, 
        train_loader: torch.utils.data.DataLoader, 
        epoch: int, 
        global_step: int, 
        logger: Logger,
        frozen: tuple[nn.Module, ...] = (),
    ):
    model.train()
    for m in frozen:  # frozen submodules stay in eval mode (no BatchNorm running-stat updates)
        m.eval()
    n_steps = len(train_loader)
    if args.training_args.max_train_steps_per_epoch is not None:
        n_steps = min(n_steps, args.training_args.max_train_steps_per_epoch)

    pbar = tqdm(train_loader, total=n_steps, desc=f"epoch {epoch}")
    for i, batch in enumerate(pbar):
        if i >= n_steps:
            break
        batch = to_device(batch, args.device, args.env_args.image_size)
        with autocast(args.device, args.training_args.bf16):
            output = lejepa_forward(batch, model, sigreg, args)

        optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.training_args.gradient_clip_val)
        optimizer.step()
        scheduler.step()
        global_step += 1

        losses = {k: v.item() for k, v in output.items() if "loss" in k}
        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        if global_step % args.training_args.log_every == 0:
            logger.log({f"train/{k}": v for k, v in losses.items()} | {"lr": scheduler.get_last_lr()[0]}, global_step)
    return global_step


@torch.no_grad()
def validate(
    args: Args, 
    model: JEPA, 
    sigreg: SIGReg, 
    val_loader: torch.utils.data.DataLoader
):
    model.eval()
    totals, n = {}, 0
    for batch in val_loader:
        batch = to_device(batch, args.device, args.env_args.image_size)
        with autocast(args.device, args.training_args.bf16):
            output = lejepa_forward(batch, model, sigreg, args)
        for k, v in output.items():
            if "loss" in k:
                totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}

def make_collate_fn():
    def collate_fn(batch):
        # each item is one episode with (W, T, ...) windows; concat episodes along W
        return {k: torch.cat([b[k] for b in batch], dim=0) for k in batch[0]}

    return collate_fn

def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def train(args: Args):
    t = args.training_args
    assert t.batch_size % t.episodes_per_batch == 0, "batch_size must be divisible by episodes_per_batch"
    if args.exp_name is None:
        args.exp_name = f"{args.env_args.env_id}__lewm__{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = ManiSkillTrajectoryDataset(
        args.env_args.demo_dataset_path,
        num_steps=args.model_args.history_size + args.model_args.predictor.num_preds,
        frameskip=args.env_args.frameskip,
        num_windows=t.batch_size // t.episodes_per_batch,
        load_count=args.env_args.max_trajectories,
    )

    # split by episode
    rnd_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        lengths=(args.env_args.train_split, 1 - args.env_args.train_split),
        generator=rnd_gen,
    )
    print(f"{len(train_set)} train / {len(val_set)} val episodes, action_dim={dataset.action_dim}")

    # z-score normalize actions with the train split's statistics
    dataset.fit_mean_std(train_set.indices)

    # each item is one episode's worth of windows, so the DataLoader batch size is the
    # number of episodes per batch (keeps intra-batch variance within-episode)
    loader_kw = dict(
        batch_size=t.episodes_per_batch,
        num_workers=t.num_workers,
        pin_memory=t.pin_memory,
        collate_fn=make_collate_fn(),
        worker_init_fn=seed_worker,
        persistent_workers=t.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=True, generator=rnd_gen, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_kw)

    model = get_model(args, dataset.action_dim).to(args.device)
    sigreg = SIGReg(knots=args.loss_args.sigreg_knots, num_proj=args.loss_args.sigreg_num_proj).to(args.device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = get_optimizer(args, model)
    steps_per_epoch = len(train_loader)
    if t.max_train_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, t.max_train_steps_per_epoch)
    scheduler = get_scheduler(args, optimizer, t.max_epochs * steps_per_epoch)

    logger = Logger(
        args.wandb_config.wandb_project_name, 
        args.wandb_config.wandb_team, 
        args.exp_name,
        asdict(args),
        log_to_wandb = args.wandb_config.enable
    )
    
    global_step = 0
    for epoch in range(t.max_epochs):
        global_step = do_epoch(args, model, sigreg, optimizer, scheduler, train_loader, epoch, global_step, logger)
        val_losses = validate(args, model, sigreg, val_loader)
        print(f"epoch {epoch} val: " + ", ".join(f"{k}={v:.4f}" for k, v in val_losses.items()))
        logger.log({f"val/{k}": v for k, v in val_losses.items()} | {"epoch": epoch}, global_step)

        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                global_step=global_step,
                action_mean=dataset.action_mean,
                action_std=dataset.action_std,
                args=asdict(args),
            ),
            os.path.join(run_dir, "latest.pt"),
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    train(tyro.cli(Args))
