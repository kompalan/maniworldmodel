"""
GMM action prior: a full-covariance Gaussian mixture over windows of action chunks.

Same data and interface as lewm.action_prior (the DDIM prior), so the two are interchangeable
in the planner. A sample is one window of `horizon` action chunks, each chunk being `frameskip`
raw actions concatenated: (horizon, frameskip * action_dim), z-scored with the train-split
statistics stored in the checkpoint.

Plans longer than one window are drawn chunk by chunk: every window conditions on the last
horizon - 1 chunks (executed context + chunks generated so far) with the closed-form Gaussian
conditional of each mixture component, and the component posteriors are re-weighted by how
well each one explains those known chunks.

Fit (from examples/baselines; CPU, a minute or two):
    lewm/.venv/bin/python -m lewm.gmm_prior --exp-name gmm_prior

Use:
    prior = GMMPrior.from_checkpoint("runs/gmm_prior/gmm_prior.pt", device="cuda")
    plans = prior.sample_continuation(context, num_chunks=13, n=1000)  # (1000, 13, fs * A), normalized
"""
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import tyro
from sklearn.mixture import GaussianMixture

from .action_prior import nn_dist
from .dataset import ManiSkillTrajectoryDataset


@dataclass
class GMMPriorArgs:
    exp_name: Optional[str] = None
    output_dir: str = "runs"
    """checkpoint goes to <output_dir>/<exp_name>/gmm_prior.pt"""
    seed: int = 3072

    demo_dataset_path: str = "/home/anurag/.maniskill/demos/PushCube-v1/cam_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    max_trajectories: Optional[int] = None
    train_split: float = 0.8
    """episode split; same seed + split as lewm.train gives the same train episodes"""
    frameskip: int = 5
    """raw actions per chunk (match the world model's frameskip)"""
    horizon: int = 4
    """chunks per window. Longer plans are chained window by window (see sample_continuation)"""

    n_components: int = 32
    reg_covar: float = 1e-3
    """added to every covariance diagonal (normalized units). Actions inside a chunk are nearly
    collinear, so this keeps the covariances well conditioned"""
    max_iter: int = 300
    n_init: int = 2
    max_fit_windows: int = 200_000
    """subsample at most this many train windows for EM"""
    num_eval_samples: int = 2000


class GMMPrior:
    """mixture parameters + the action normalization they were fit with."""

    def __init__(self, weights, means, covariances, horizon: int, action_mean, action_std, frameskip: int, device="cpu"):
        # keep the mixture in float64: conditioning inverts near-singular covariance blocks
        self.weights = torch.as_tensor(weights, dtype=torch.float64, device=device)  # (C,)
        self.means = torch.as_tensor(means, dtype=torch.float64, device=device)  # (C, D)
        self.covariances = torch.as_tensor(covariances, dtype=torch.float64, device=device)  # (C, D, D)
        self.horizon = horizon
        self.chunk_dim = self.means.shape[1] // horizon
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)  # (A,) per raw action
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.frameskip = frameskip
        self._cond_cache = {}

    @property
    def device(self):
        return self.means.device

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            ckpt["weights"], ckpt["means"], ckpt["covariances"], ckpt["args"]["horizon"],
            ckpt["action_mean"], ckpt["action_std"], ckpt["args"]["frameskip"], device=device,
        )

    def _conditional(self, k: int):
        """per-component Gaussian conditioning on the first k chunks (cached per k).

        Returns the marginal of the known part (for re-weighting components), the gain
        S_uk S_kk^-1 and the Cholesky factor of the conditional covariance of the unknown part.
        """
        if k not in self._cond_cache:
            K = k * self.chunk_dim
            mu, S = self.means, self.covariances
            if K == 0:
                self._cond_cache[k] = (None, None, mu, torch.linalg.cholesky(S))
            else:
                S_kk, S_uk, S_uu = S[:, :K, :K], S[:, K:, :K], S[:, K:, K:]
                gain = torch.linalg.solve(S_kk, S_uk.transpose(1, 2)).transpose(1, 2)  # (C, U, K)
                cov = S_uu - gain @ S_uk.transpose(1, 2)
                cov = 0.5 * (cov + cov.transpose(1, 2))
                marginal = torch.distributions.MultivariateNormal(mu[:, :K], covariance_matrix=S_kk)
                self._cond_cache[k] = (marginal, gain, mu[:, K:], torch.linalg.cholesky(cov))
        return self._cond_cache[k]

    def _randn(self, shape, generator: Optional[torch.Generator]):
        device = generator.device if generator is not None else self.device
        return torch.randn(shape, generator=generator, device=device).to(self.device, torch.float64)

    def _complete(self, known: torch.Tensor, n: int, generator) -> torch.Tensor:
        """sample n windows whose first k chunks are `known` (n, k, chunk_dim), k may be 0.
        Returns the whole window (n, horizon, chunk_dim), float64."""
        k = known.shape[1]
        marginal, gain, mu_u, chol = self._conditional(k)
        log_w = self.weights.log().expand(n, -1)
        x_k = known.reshape(n, -1).to(self.device, torch.float64)
        if k:
            log_w = log_w + marginal.log_prob(x_k[:, None, :])  # (n, C) posterior over components
        u = torch.rand(n, 1, generator=generator, device=generator.device if generator is not None else self.device)
        cdf = log_w.softmax(-1).cumsum(-1)
        c = torch.searchsorted(cdf, u.to(self.device, torch.float64)).clamp_max(len(self.weights) - 1).squeeze(1)
        mean = mu_u[c]
        if k:
            mean = mean + (gain[c] @ (x_k - self.means[c, : x_k.shape[1]])[..., None]).squeeze(-1)
        eps = self._randn((n, chol.shape[-1]), generator)
        x_u = mean + (chol[c] @ eps[..., None]).squeeze(-1)
        return torch.cat([x_k, x_u], dim=1).view(n, self.horizon, self.chunk_dim)

    @torch.no_grad()
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """n normalized action windows, (n, horizon, frameskip * A)."""
        known = torch.empty(n, 0, self.chunk_dim, device=self.device)
        return self._complete(known, n, generator).float()

    @torch.no_grad()
    def sample_continuation(
        self, context: torch.Tensor, num_chunks: int, n: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """n continuations of `num_chunks` chunks after `context`, rolled out window by window.

        context: (C, frameskip * A) already-executed chunks, normalized with this prior's stats.
        Returns (n, num_chunks, frameskip * A), normalized.
        """
        k = min(context.shape[0], self.horizon - 1)
        seq = context[context.shape[0] - k :].to(self.device, torch.float64).expand(n, k, -1)
        start = k
        while seq.shape[1] - start < num_chunks:
            k_now = min(seq.shape[1], self.horizon - 1)
            window = self._complete(seq[:, seq.shape[1] - k_now :], n, generator)
            seq = torch.cat([seq, window[:, k_now:]], dim=1)
        return seq[:, start : start + num_chunks].float()

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """(..., C * frameskip, A) raw env actions -> (..., C, frameskip * A) normalized."""
        x = (raw - self.action_mean.to(raw.device)) / self.action_std.to(raw.device)
        return x.reshape(*raw.shape[:-2], -1, self.chunk_dim)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """(..., horizon, frameskip * A) normalized -> (..., horizon * frameskip, A) raw env actions."""
        A = self.action_mean.numel()
        raw = x.reshape(*x.shape[:-2], -1, A)
        return raw * self.action_std.to(x.device) + self.action_mean.to(x.device)


def fit(args: GMMPriorArgs):
    if args.exp_name is None:
        args.exp_name = f"gmm_prior__{args.seed}__{int(time.time())}"
    run_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

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
    train_x = dataset.all_action_windows(train_set.indices)  # (N, horizon, fs * A)
    val_x = dataset.all_action_windows(val_set.indices)
    print(f"{len(train_set)} train / {len(val_set)} val episodes -> {len(train_x)} / {len(val_x)} windows of {train_x.shape[1:]}")

    rng = np.random.default_rng(args.seed)
    fit_x = train_x.reshape(len(train_x), -1).astype(np.float64)
    if len(fit_x) > args.max_fit_windows:
        fit_x = fit_x[rng.choice(len(fit_x), args.max_fit_windows, replace=False)]
    val_flat = val_x.reshape(len(val_x), -1).astype(np.float64)

    t0 = time.time()
    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type="full",
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=args.n_init,
        random_state=args.seed,
        verbose=1,
    ).fit(fit_x)
    metrics = {
        "converged": bool(gmm.converged_),
        "train_loglik": float(gmm.score(fit_x)),
        "val_loglik": float(gmm.score(val_flat)),
        "bic": float(gmm.bic(fit_x)),
    }
    print(f"fit in {time.time() - t0:.1f}s: {metrics}")

    prior = GMMPrior(gmm.weights_, gmm.means_, gmm.covariances_, args.horizon,
                     dataset.action_mean, dataset.action_std, args.frameskip)
    n = min(args.num_eval_samples, len(val_x))
    samples = prior.sample(n, torch.Generator().manual_seed(args.seed)).flatten(1)
    tr, va = torch.from_numpy(train_x).flatten(1), torch.from_numpy(val_x[:n]).flatten(1)
    metrics.update({
        "ref/val_to_train_nn": nn_dist(va, tr).item(),
        "gmm/precision_nn": nn_dist(samples, tr).item(),
        "gmm/recall_nn": nn_dist(va, samples).item(),
    })
    print(", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not isinstance(v, bool)))

    path = os.path.join(run_dir, "gmm_prior.pt")
    torch.save(
        dict(
            weights=torch.from_numpy(gmm.weights_),
            means=torch.from_numpy(gmm.means_),
            covariances=torch.from_numpy(gmm.covariances_),
            action_mean=dataset.action_mean,
            action_std=dataset.action_std,
            args=asdict(args),
            metrics=metrics,
        ),
        path,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    fit(tyro.cli(GMMPriorArgs))
