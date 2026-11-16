"""
Online LeWM training: alternate between
  1. collecting `rollouts_per_round` planner rollouts (GMM-seeded CEM, receding horizon 1) on
     train-split tasks and adding them to a RolloutBuffer (evicting unpinned expert demos first)
  2. `grad_steps_per_round` gradient steps on windows sampled from the buffer
Validation stays on the fixed expert val split, so it is comparable to train.py's.

A task is a train-split demo's (env seed, final frame as the goal). Tasks are kept separately
from the buffer, so they survive the demo being evicted.

Run from examples/baselines:
    lewm/.venv/bin/python -m lewm.online --rollouts-per-round 8 --grad-steps-per-round 200
"""
import os
import time
from dataclasses import asdict

import dotenv
import numpy as np
import torch
import tyro
from tqdm import tqdm

from .config import Args, EvalArgs, OnlineArgs
from .dataset import ManiSkillTrajectoryDataset
from .env.evaluate import _base_env, _step, hold_action
from .env.make_env import make_eval_envs
from .eval import cem_plan_with_gmm_prior
from .gmm_prior import GMMPrior
from .rollout_buffer import RolloutBuffer
from .train import do_epoch, get_optimizer, make_collate_fn, seed_worker, validate
from .module import SIGReg
from .utils import Logger, _dataclass_from_dict, get_model


@torch.no_grad()
def collect_rollout(envs, plan_fn, env_seed: int, H: int, fs: int, o: OnlineArgs, demo_prefix=None):
    """one episode from reset. Returns (obs (L+1, h, w, 3) uint8, raw actions (L, A), success)."""
    obs, _ = envs.reset(seed=env_seed)
    frames, actions = [torch.as_tensor(obs)[0, -1]], []

    def step(chunk):
        nonlocal obs
        for a in chunk:
            obs, *_ = _step(envs, a)
            frames.append(torch.as_tensor(obs)[0, -1])
            actions.append(torch.as_tensor(a, dtype=torch.float32))
            if bool(_base_env(envs).evaluate()["success"][0]):
                return True
        return False

    if demo_prefix is None:
        hold = hold_action(envs)
        if o.hold_gripper is not None:
            hold[-1] = o.hold_gripper
        prefix = hold.repeat(H * fs, 1)
    else:
        prefix = demo_prefix
    success = step(prefix)
    while not success and len(actions) + fs <= o.max_episode_steps:
        success = step(plan_fn(torch.as_tensor(obs), torch.stack(actions)))
    return torch.stack(frames).numpy(), torch.stack(actions).numpy(), success


def load_init(o: OnlineArgs) -> tuple[Args, dict]:
    """the init checkpoint and its training Args, with this machine's overrides."""
    ckpt = torch.load(o.init_ckpt, map_location="cpu", weights_only=False)
    args = _dataclass_from_dict(Args, ckpt["args"])
    args.device = o.device
    args.env_args.demo_dataset_path = o.demo_dataset_path
    if o.max_trajectories is not None:
        args.env_args.max_trajectories = o.max_trajectories
    args.training_args.max_train_steps_per_epoch = None  # do_epoch runs the whole round's loader
    return args, ckpt


def build_data(o: OnlineArgs, args: Args, ckpt: dict):
    """expert demos -> (dataset, buffer of the train split, tasks, expert val loader).
    Split exactly like train.py, so val is the pretraining val split."""
    t, H, fs = args.training_args, args.model_args.history_size, args.env_args.frameskip
    ds = ManiSkillTrajectoryDataset(
        args.env_args.demo_dataset_path,
        num_steps=H + args.model_args.predictor.num_preds,
        frameskip=fs,
        num_windows=t.batch_size // t.episodes_per_batch,
        load_count=args.env_args.max_trajectories,
    )
    train_set, val_set = torch.utils.data.random_split(
        ds, (args.env_args.train_split, 1 - args.env_args.train_split),
        generator=torch.Generator().manual_seed(args.seed),
    )
    ds.action_mean, ds.action_std = ckpt["action_mean"], ckpt["action_std"]
    buf = RolloutBuffer.from_dataset(ds, train_set.indices, ds.num_windows, o.pin_frac, seed=o.seed)
    print(f"buffer: {len(buf)} expert episodes, {int(buf.pinned.sum())} pinned")

    # (env seed, goal = final frame, raw demo prefix) of every train-split demo
    tasks = []
    for i in train_set.indices:
        o0, a0, _ = ds.episode_windows[i]
        L = ds.episodes[i]["elapsed_steps"]
        tasks.append((ds.episodes[i]["episode_seed"], torch.tensor(ds.obs[o0 + L]),
                      torch.tensor(ds.actions[a0 : a0 + H * fs])))

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=t.episodes_per_batch, shuffle=False, num_workers=o.num_workers,
        collate_fn=make_collate_fn(), worker_init_fn=seed_worker, persistent_workers=o.num_workers > 0,
    )
    return ds, buf, tasks, val_loader


def frozen_modules(o: OnlineArgs, model) -> tuple[torch.nn.Module, ...]:
    """submodules kept frozen (no gradients, eval mode so BatchNorm running stats never update)."""
    mods = [model.encoder, model.projector] if o.freeze_encoder else []
    if o.freeze_bn:
        mods += [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    return tuple(mods)


def build_model(o: OnlineArgs, args: Args, ckpt: dict, action_dim: int):
    """model + SIGReg + optimizer from the checkpoint, and a constant-LR scheduler for do_epoch."""
    model = get_model(args, action_dim)
    model.load_state_dict(ckpt["model"])
    model.to(o.device)
    for m in frozen_modules(o, model):
        m.requires_grad_(False)
    sigreg = SIGReg(knots=args.loss_args.sigreg_knots, num_proj=args.loss_args.sigreg_num_proj).to(o.device)
    optimizer = get_optimizer(args, model)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    # not the saved optimizer's lr: train.py's cosine schedule leaves it at 0
    lr = o.learning_rate if o.learning_rate is not None else args.training_args.optimizer_args.learning_rate
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] = lr
    print(f"online learning rate: {lr:g}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, sigreg, optimizer, scheduler


def make_planner(o: OnlineArgs, args: Args, ckpt: dict, model):
    """returns plan_fn_for(goal) -> plan_fn(obs, actions_so_far): GMM-seeded CEM, one chunk per call."""
    prior = GMMPrior.from_checkpoint(o.gmm_prior_ckpt, device=o.device)
    eval_args = EvalArgs(
        exp_name="online", ckpt_path=o.init_ckpt, plan_horizon=o.plan_horizon, plan_receding_horizon=1,
        prior_num_samples=o.prior_num_samples, prior_cem_iters=o.prior_cem_iters,
    )
    am = torch.as_tensor(ckpt["action_mean"], dtype=torch.float32)
    sd = torch.as_tensor(ckpt["action_std"], dtype=torch.float32)
    gen = torch.Generator(device=o.device).manual_seed(o.seed)

    def plan_fn_for(goal):
        return lambda obs, acts: cem_plan_with_gmm_prior(
            args, eval_args, model, prior, goal, am, sd, obs, acts, generator=gen)
    return plan_fn_for


def make_envs(o: OnlineArgs, args: Args, ds: ManiSkillTrajectoryDataset):
    H, fs = args.model_args.history_size, args.env_args.frameskip
    return make_eval_envs(
        env_id=args.env_args.env_id, num_envs=1, sim_backend="physx_cpu",
        # +1 so collect_rollout's own budget ends the episode before the env truncates (and autoresets)
        env_kwargs={**ds.env_info["env_kwargs"], "max_episode_steps": o.max_episode_steps + 1},
        other_kwargs={"obs_horizon": H + 1, "frameskip": fs},
    )


def collect(o: OnlineArgs, args: Args, envs, buf: RolloutBuffer, tasks, plan_fn_for, rng, rnd: int) -> dict:
    """rollouts_per_round planner rollouts on random tasks, each added to the buffer."""
    H, fs = args.model_args.history_size, args.env_args.frameskip
    succ, lens = [], []
    for k in tqdm(rng.choice(len(tasks), o.rollouts_per_round), desc=f"round {rnd} collect"):
        env_seed, goal, demo_prefix = tasks[k]
        obs, acts, s = collect_rollout(envs, plan_fn_for(goal), env_seed, H, fs, o,
                                       demo_prefix if o.seed_mode == "demo" else None)
        buf.add(obs, acts, env_seed)
        succ.append(s)
        lens.append(len(acts))
    return {"collect/success": float(np.mean(succ)), "collect/episode_len": float(np.mean(lens))}


def buffer_loader(o: OnlineArgs, args: Args, buf: RolloutBuffer):
    """grad_steps_per_round batches of episodes sampled with replacement. Build it after every
    collect: its workers fork the buffer as it is at that point."""
    epb = args.training_args.episodes_per_batch
    return torch.utils.data.DataLoader(
        buf, batch_size=epb, num_workers=o.num_workers, collate_fn=make_collate_fn(),
        worker_init_fn=seed_worker, pin_memory=args.training_args.pin_memory, drop_last=True,
        sampler=torch.utils.data.RandomSampler(buf, replacement=True, num_samples=o.grad_steps_per_round * epb),
    )


@torch.no_grad()
def perturb_toward_init(model, args: Args, action_dim: int, alpha: float, seed: int):
    """theta <- (1 - alpha) theta + alpha theta_rand, with theta_rand a fresh get_model init.
    Only parameters move; buffers (BatchNorm running stats) and optimizer state are kept."""
    with torch.random.fork_rng(devices=[]):  # don't disturb the global RNG stream
        torch.manual_seed(seed)
        fresh = get_model(args, action_dim)
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        p.lerp_(q.to(p.device), alpha)


def save(path: str, o: OnlineArgs, args: Args, ckpt: dict, model, optimizer, rnd: int, step: int):
    """same format as train.py's checkpoint, so eval.py loads it."""
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=rnd, global_step=step,
                    action_mean=ckpt["action_mean"], action_std=ckpt["action_std"], args=asdict(args),
                    online_args=asdict(o)), path)


def main(o: OnlineArgs):
    assert o.seed_mode in ("hold", "demo"), o.seed_mode
    torch.manual_seed(o.seed)
    np.random.seed(o.seed)
    exp_name = o.exp_name or f"online__{int(time.time())}"
    run_dir = os.path.join(o.output_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    args, ckpt = load_init(o)
    ds, buf, tasks, val_loader = build_data(o, args, ckpt)
    model, sigreg, optimizer, scheduler = build_model(o, args, ckpt, ds.action_dim)
    plan_fn_for = make_planner(o, args, ckpt, model)
    envs = make_envs(o, args, ds)
    logger = Logger(o.wandb_config.wandb_project_name, o.wandb_config.wandb_team, exp_name,
                    asdict(o) | {"init_args": asdict(args)}, o.wandb_config.enable)
    rng = np.random.default_rng(o.seed)

    step = 0
    for rnd in range(o.rounds):
        model.eval()  # BatchNorm on running stats while planning, as in eval.py
        metrics = collect(o, args, envs, buf, tasks, plan_fn_for, rng, rnd)
        if o.perturb_alpha > 0:
            perturb_toward_init(model, args, ds.action_dim, o.perturb_alpha, seed=o.seed * 100_000 + rnd)
        step = do_epoch(args, model, sigreg, optimizer, scheduler, buffer_loader(o, args, buf), rnd, step, logger,
                        frozen=frozen_modules(o, model))

        metrics |= {"round": rnd} | {f"buffer/{k}": v for k, v in buf.stats().items()}
        if (rnd + 1) % o.val_every == 0 or rnd == o.rounds - 1:
            metrics |= {f"val/{k}": v for k, v in validate(args, model, sigreg, val_loader).items()}
            save(os.path.join(run_dir, "latest.pt"), o, args, ckpt, model, optimizer, rnd, step)
        print(f"round {rnd}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                           for k, v in metrics.items()))
        logger.log(metrics, step)
    envs.close()


if __name__ == "__main__":
    dotenv.load_dotenv()
    main(tyro.cli(OnlineArgs))
