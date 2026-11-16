"""
Unconditional action prior: a DDIM model of p(action window) on ManiSkillTrajectoryDataset.

A sample is one window of `horizon` action chunks, each chunk being `frameskip` raw actions
concatenated: shape (horizon, frameskip * action_dim). Samples are in the same z-scored units
as the dataset's `action` field (and as the CEM planner in eval.py), using the dataset
statistics stored in the checkpoint.

Train (from examples/baselines):
    lewm/.venv/bin/python -m lewm.action_prior --exp-name action_prior

Use:
    prior = ActionPrior.from_checkpoint("runs/action_prior/action_prior.pt", device="cuda")
    actions = prior.sample(300, num_inference_steps=10)  # (300, horizon, frameskip * A), normalized
"""
import copy
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from diffusers import DDIMScheduler
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.training_utils import EMAModel
from tqdm import tqdm

from .dataset import ManiSkillTrajectoryDataset


@dataclass
class ActionPriorArgs:
    exp_name: Optional[str] = None
    output_dir: str = "runs"
    """checkpoint goes to <output_dir>/<exp_name>/action_prior.pt"""
    device: str = "cuda"
    seed: int = 3072

    demo_dataset_path: str = "/home/anurag/.maniskill/demos/PushCube-v1/cam_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    max_trajectories: Optional[int] = None
    train_split: float = 0.8
    """episode split; same seed + split as lewm.train gives the same train episodes"""
    frameskip: int = 5
    """raw actions per chunk (match the world model's frameskip)"""
    horizon: int = 4
    """chunks per sampled window. Episodes shorter than horizon * frameskip env steps are skipped"""

    hidden_dim: int = 512
    num_blocks: int = 4
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"

    batch_size: int = 256
    train_steps: int = 20_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    ema_decay: float = 0.999
    log_every: int = 500
    eval_every: int = 5000

    num_inference_steps: int = 10
    """DDIM steps used for the eval samples"""
    num_eval_samples: int = 2000


class ResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)  # scale/shift from the timestep embedding
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class ActionDenoiser(nn.Module):
    """epsilon-prediction MLP over a flattened action window."""

    def __init__(self, horizon: int, chunk_dim: int, hidden_dim: int = 512, num_blocks: int = 4):
        super().__init__()
        self.horizon, self.chunk_dim = horizon, chunk_dim
        in_dim = horizon * chunk_dim
        self.time_proj = Timesteps(hidden_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_emb = TimestepEmbedding(hidden_dim, hidden_dim)
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, in_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        """x: (B, horizon, chunk_dim) noisy actions, t: (B,) timesteps -> predicted noise like x."""
        cond = self.time_emb(self.time_proj(t).to(x.dtype))
        h = self.inp(x.flatten(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).view_as(x)


def make_scheduler(args: ActionPriorArgs) -> DDIMScheduler:
    # actions are z-scored, not in [-1, 1], so no clipping
    return DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
        prediction_type="epsilon",
    )


class ActionPrior:
    """denoiser + DDIM scheduler + the action normalization it was trained with."""

    def __init__(self, model: ActionDenoiser, scheduler: DDIMScheduler, action_mean, action_std, frameskip: int):
        self.model = model.eval()
        self.scheduler = scheduler
        self.action_mean = torch.as_tensor(action_mean)  # (A,) per raw action
        self.action_std = torch.as_tensor(action_std)
        self.frameskip = frameskip

    @property
    def device(self):
        return next(self.model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = ActionPriorArgs(**ckpt["args"])
        model = ActionDenoiser(args.horizon, ckpt["chunk_dim"], args.hidden_dim, args.num_blocks)
        model.load_state_dict(ckpt["model"])
        return cls(model.to(device), make_scheduler(args), ckpt["action_mean"], ckpt["action_std"], args.frameskip)

    def _randn(self, shape, generator: Optional[torch.Generator]):
        device = generator.device if generator is not None else self.device
        return torch.randn(shape, generator=generator, device=device).to(self.device)

    @torch.no_grad()
    def _inpaint(self, known: torch.Tensor, n: int, num_inference_steps: int, eta: float, generator):
        """DDIM-sample n windows whose first k chunks are `known` (n, k, D) (k may be 0).

        Replacement inpainting: before every denoising step the known chunks are overwritten
        with a sample of q(x_t | known), so the generated chunks are pulled towards
        continuations that are consistent with them.
        """
        # a private scheduler copy: set_timesteps mutates it, so concurrent callers don't interfere
        scheduler = copy.deepcopy(self.scheduler)
        scheduler.set_timesteps(num_inference_steps)
        k = known.shape[1]
        x = self._randn((n, self.model.horizon, self.model.chunk_dim), generator)
        for t in scheduler.timesteps:
            tt = t.expand(n).to(self.device)
            if k:
                x[:, :k] = scheduler.add_noise(known, self._randn(known.shape, generator), tt)
            eps = self.model(x, tt)
            x = scheduler.step(eps, t, x, eta=eta, generator=generator).prev_sample
        if k:
            x[:, :k] = known
        return x

    @torch.no_grad()
    def sample(self, n: int, num_inference_steps: int = 10, eta: float = 0.0, generator: Optional[torch.Generator] = None):
        """n normalized action windows, (n, horizon, frameskip * A). eta=0 is deterministic DDIM."""
        known = torch.empty(n, 0, self.model.chunk_dim, device=self.device)
        return self._inpaint(known, n, num_inference_steps, eta, generator)

    @torch.no_grad()
    def sample_continuation(
        self,
        context: torch.Tensor,
        num_chunks: int,
        n: int,
        num_inference_steps: int = 10,
        eta: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """n continuations of `num_chunks` chunks after `context`, rolled out window by window.

        context: (C, frameskip * A) already-executed chunks, normalized with this prior's stats.
        Each window conditions on the last min(C, horizon - 1) chunks (context + generated so far)
        and generates the rest, so samples stay on the action manifold and start from the
        current state. Returns (n, num_chunks, frameskip * A), normalized.
        """
        k = min(context.shape[0], self.model.horizon - 1)
        seq = context[-k:].to(self.device).expand(n, k, -1) if k else torch.empty(n, 0, self.model.chunk_dim, device=self.device)
        start = seq.shape[1]
        while seq.shape[1] - start < num_chunks:
            window = self._inpaint(seq[:, seq.shape[1] - k :], n, num_inference_steps, eta, generator)
            seq = torch.cat([seq, window[:, k:]], dim=1)
        return seq[:, start : start + num_chunks]

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """(..., C * frameskip, A) raw env actions -> (..., C, frameskip * A) normalized."""
        x = (raw - self.action_mean.to(raw.device)) / self.action_std.to(raw.device)
        return x.reshape(*raw.shape[:-2], -1, self.model.chunk_dim)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """(..., horizon, frameskip * A) normalized -> (..., horizon * frameskip, A) raw env actions."""
        A = self.action_mean.numel()
        raw = x.reshape(*x.shape[:-2], -1, A)
        return raw * self.action_std.to(x.device) + self.action_mean.to(x.device)


def nn_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """mean Euclidean distance from each row of a to its nearest row of b."""
    out = [torch.cdist(a[i : i + chunk], b).min(dim=1).values for i in range(0, len(a), chunk)]
    return torch.cat(out).mean()


@torch.no_grad()
def evaluate(prior: ActionPrior, train_x: torch.Tensor, val_x: torch.Tensor, args: ActionPriorArgs) -> dict:
    """compare samples against held-out windows, with a full-covariance Gaussian as baseline.

    precision: samples -> nearest train window (are samples on the data manifold?)
    recall:    val windows -> nearest sample (do samples cover held-out data?)
    lower is better for all; val->train precision is the "real data" reference.
    """
    n = min(args.num_eval_samples, len(val_x))
    g = torch.Generator(device=prior.device).manual_seed(args.seed)
    samples = prior.sample(n, args.num_inference_steps, generator=g).flatten(1)
    tr, va = train_x.flatten(1), val_x[:n].flatten(1)

    mu, cov = tr.mean(0), torch.cov(tr.T)
    gauss = torch.distributions.MultivariateNormal(mu, cov + 1e-4 * torch.eye(len(mu), device=mu.device)).sample((n,))

    m = {"ref/val_to_train_nn": nn_dist(va, tr).item()}
    for name, s in [("ddim", samples), ("gaussian", gauss)]:
        m[f"{name}/precision_nn"] = nn_dist(s, tr).item()
        m[f"{name}/recall_nn"] = nn_dist(va, s).item()
        m[f"{name}/mean_err"] = (s.mean(0) - va.mean(0)).abs().mean().item()
        m[f"{name}/std_err"] = (s.std(0) - va.std(0)).abs().mean().item()
    return m


def train(args: ActionPriorArgs):
    if args.exp_name is None:
        args.exp_name = f"action_prior__{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = ManiSkillTrajectoryDataset(
        args.demo_dataset_path,
        num_steps=args.horizon,
        frameskip=args.frameskip,
        load_count=args.max_trajectories,
        load_pixels=False,
    )
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        lengths=(args.train_split, 1 - args.train_split),
        generator=torch.Generator().manual_seed(args.seed),
    )
    dataset.fit_mean_std(train_set.indices)
    # actions are tiny, so keep every window on the device and sample windows uniformly
    train_x = torch.from_numpy(dataset.all_action_windows(train_set.indices)).to(args.device)
    val_x = torch.from_numpy(dataset.all_action_windows(val_set.indices)).to(args.device)
    print(f"{len(train_set)} train / {len(val_set)} val episodes -> {len(train_x)} / {len(val_x)} windows of {tuple(train_x.shape[1:])}")

    model = ActionDenoiser(args.horizon, train_x.shape[-1], args.hidden_dim, args.num_blocks).to(args.device)
    print(f"denoiser params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ema = EMAModel(model.parameters(), decay=args.ema_decay)
    scheduler = make_scheduler(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_steps)

    def ema_prior():
        ema_model = copy.deepcopy(model)
        ema.copy_to(ema_model.parameters())
        return ActionPrior(ema_model, scheduler, dataset.action_mean, dataset.action_std, args.frameskip)

    def save(prior: ActionPrior, metrics: dict):
        torch.save(
            dict(
                model=prior.model.state_dict(),
                chunk_dim=train_x.shape[-1],
                action_mean=dataset.action_mean,
                action_std=dataset.action_std,
                args=asdict(args),
                metrics=metrics,
            ),
            os.path.join(run_dir, "action_prior.pt"),
        )

    model.train()
    running = 0.0
    pbar = tqdm(range(1, args.train_steps + 1))
    for step in pbar:
        x0 = train_x[torch.randint(len(train_x), (args.batch_size,), device=args.device)]
        noise = torch.randn_like(x0)
        t = torch.randint(args.num_train_timesteps, (args.batch_size,), device=args.device)
        xt = scheduler.add_noise(x0, noise, t)
        loss = F.mse_loss(model(xt, t), noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()
        ema.step(model.parameters())

        running += loss.item()
        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{running / args.log_every:.4f}")
            running = 0.0
        if step % args.eval_every == 0 or step == args.train_steps:
            prior = ema_prior()
            metrics = evaluate(prior, train_x, val_x, args)
            tqdm.write(f"step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            save(prior, metrics)
    print(f"saved {os.path.join(run_dir, 'action_prior.pt')}")


if __name__ == "__main__":
    train(tyro.cli(ActionPriorArgs))
