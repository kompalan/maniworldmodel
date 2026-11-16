"""
How does the best cost CEM finds compare with the cost of the ground-truth (demo) action sequence?

For each held-out episode and seed mode, this sets up the planner's first call exactly like eval.py
and runs two planners on the same context:

  raw CEM   Gaussian CEM from "hold the last action" (cem_plan), cem_iters rounds
  GMM CEM   best of prior_num_samples GMM seeds (iteration 0), then CEM from their elites
            (cem_plan_with_gmm_prior), gmm_iters rounds

and records, per iteration, the lowest world-model cost found so far. The reference is the cost of
the demo's own actions over the same horizon (from step 0 after a hold prefix, from step H*fs after
a demo prefix), padded by holding the demo's last action.

  gap = best cost so far - true plan cost      (< 0: CEM found a plan the model rates better)

Writes <out_dir>/cem_convergence.json and cem_convergence.png (one line chart per planner).

Run from examples/baselines:
    lewm/.venv/bin/python -m lewm.cem_convergence --ckpt-path lewm/runs/lewm_bn_ep8/latest.pt
    lewm/.venv/bin/python -m lewm.cem_convergence --plot-only --out-dir <dir>   # redraw the charts
"""
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import tyro

from .config import Args, EvalArgs
from .dataset import ManiSkillTrajectoryDataset
from .env.evaluate import _step, hold_action
from .env.make_env import make_eval_envs
from .eval import _plan_context, _score
from .gmm_prior import GMMPrior
from .utils import _dataclass_from_dict, get_model


@dataclass
class ConvergenceArgs:
    ckpt_path: str = "lewm/runs/lewm_bn_ep8/latest.pt"
    gmm_prior_ckpt: str = "runs/gmm_prior/gmm_prior.pt"
    heldout_dataset: str = EvalArgs.heldout_dataset
    out_dir: str = "runs/cem_convergence"
    num_episodes: int = 20
    modes: tuple[str, ...] = ("hold", "demo")
    plan_horizon: int = EvalArgs.plan_horizon
    cem_num_samples: int = EvalArgs.cem_num_samples
    cem_topk: int = EvalArgs.cem_topk
    cem_var_scale: float = EvalArgs.cem_var_scale
    cem_iters: int = EvalArgs.cem_iters
    """raw-CEM rounds (eval.py default)"""
    gmm_iters: int = 30
    """CEM rounds after GMM seeding (eval.py's prior_cem_iters default is 10; run longer to see the plateau)"""
    prior_num_samples: int = EvalArgs.prior_num_samples
    hold_gripper: float | None = EvalArgs.hold_gripper
    seed: int = 0
    device: str = "cuda"
    plot_only: bool = False
    """skip planning and redraw the charts from <out_dir>/cem_convergence.json"""


@torch.no_grad()
def cem_trace(model, c: ConvergenceArgs, pixels, goal_px, hist, mean, std, iters, gen, best0=float("inf")):
    """standard CEM (same update as eval._cem); returns the best cost so far after each round."""
    S, K = c.cem_num_samples, c.cem_topk
    best, trace = best0, []
    for _ in range(iters):
        future = mean + std * torch.randn(S, *mean.shape, device=mean.device, generator=gen)
        future[0] = mean
        cost = _score(model, pixels, goal_px, hist, future, S)
        best = min(best, cost.min().item())
        trace.append(best)
        elite = future[cost.topk(K, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return trace


@torch.no_grad()
def run(c: ConvergenceArgs):
    torch.manual_seed(c.seed)
    ckpt = torch.load(c.ckpt_path, map_location="cpu", weights_only=False)
    args = _dataclass_from_dict(Args, ckpt["args"])
    H, fs, T = args.model_args.history_size, args.env_args.frameskip, c.plan_horizon
    dev = c.device

    data = ManiSkillTrajectoryDataset(c.heldout_dataset, num_steps=H + 1, frameskip=fs, load_count=c.num_episodes)
    A = data.action_dim
    model = get_model(args, A)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval().requires_grad_(False)
    am = torch.as_tensor(ckpt["action_mean"], dtype=torch.float32)
    sd = torch.as_tensor(ckpt["action_std"], dtype=torch.float32)
    prior = GMMPrior.from_checkpoint(c.gmm_prior_ckpt, device=dev)

    envs = make_eval_envs(
        env_id=args.env_args.env_id, num_envs=1, sim_backend="physx_cpu",
        env_kwargs={**data.env_info["env_kwargs"], "max_episode_steps": 100},
        other_kwargs={"obs_horizon": H + 1, "frameskip": fs},
    )

    rows = []
    for mode in c.modes:
        assert mode in ("hold", "demo"), mode
        start = 0 if mode == "hold" else H * fs  # demo step the first plan starts at
        for i, ep in enumerate(data.episodes[: c.num_episodes]):
            o0, a0, _ = data.episode_windows[i]
            goal = torch.tensor(data.obs[o0 + ep["elapsed_steps"]])
            demo = torch.tensor(data.actions[a0 : a0 + ep["elapsed_steps"]])

            # the planner's first call, as in eval.py (no seed_obs swap: the rendered frames are used)
            obs, _ = envs.reset(seed=ep["episode_seed"])
            if mode == "hold":
                act = hold_action(envs)
                if c.hold_gripper is not None:
                    act[-1] = c.hold_gripper
                executed = act.repeat(H * fs, 1)
            else:
                executed = demo[: H * fs].float()
            for a in executed:
                obs, *_ = _step(envs, a)
            obs = torch.as_tensor(obs)
            pixels, goal_px, hist = _plan_context(args, goal, am, sd, obs, executed, None, dev)
            norm = lambda raw: ((raw.to(dev) - am.to(dev)) / sd.to(dev)).reshape(-1, T, fs * A)

            # ground-truth plan over the same horizon
            d = demo[start : start + T * fs]
            true = norm(torch.cat([d, demo[-1:].expand(T * fs - len(d), -1)]))
            true_cost = _score(model, pixels, goal_px, hist, true, 1)[0].item()

            # raw CEM, initialized like cem_plan
            gen = torch.Generator(device=dev).manual_seed(c.seed * 10_000 + i)
            last = ((executed[-1] - am) / sd).to(dev)
            mean = last.repeat(T, fs)
            init_cost = _score(model, pixels, goal_px, hist, mean[None], 1)[0].item()
            std = torch.full((T, fs * A), c.cem_var_scale, device=dev)
            raw = [init_cost] + cem_trace(model, c, pixels, goal_px, hist, mean, std, c.cem_iters, gen, init_cost)

            # GMM-seeded CEM, initialized like cem_plan_with_gmm_prior
            gen = torch.Generator(device=dev).manual_seed(c.seed * 10_000 + i)
            n_exec = executed.shape[0] // fs * fs
            ctx = prior.normalize(executed[executed.shape[0] - n_exec :].to(prior.device).float())
            seeds = norm(prior.unnormalize(prior.sample_continuation(ctx, T, c.prior_num_samples, gen)))
            cost = _score(model, pixels, goal_px, hist, seeds, c.cem_num_samples)
            top = cost.topk(c.cem_topk, largest=False)
            elite = seeds[top.indices]
            best0 = cost.min().item()
            gmm = [best0] + cem_trace(
                model, c, pixels, goal_px, hist, elite.mean(0), elite.std(0), c.gmm_iters, gen, best0
            )

            rows.append(dict(mode=mode, episode=i, true_cost=true_cost, raw=raw, gmm=gmm))
            print(
                f"{mode} ep{i}: true {true_cost:.2f} | raw CEM {raw[0]:.2f} -> {raw[-1]:.2f}"
                f" | GMM {gmm[0]:.2f} -> {gmm[-1]:.2f}",
                flush=True,
            )
    envs.close()
    return rows


def summarize(rows, modes):
    """print mean gap (best so far - true cost) at a few iterations."""
    print("\n==== best-so-far cost minus true-plan cost, mean over episodes (negative = below the true plan) ====")
    for planner in ("raw", "gmm"):
        n = len(rows[0][planner])
        its = sorted({0, 1, 5, 10, 20, n - 1} & set(range(n)))
        print(f"\n{planner.upper()} CEM" + "".join(f"{f'it {k}':>16s}" for k in its))
        for mode in modes:
            rs = [r for r in rows if r["mode"] == mode]
            gap = np.array([[r[planner][k] - r["true_cost"] for k in its] for r in rs])
            below = np.array([[r[planner][k] < r["true_cost"] for k in its] for r in rs])
            print(f"  {mode:5s} gap   " + "".join(f"{g:16.2f}" for g in gap.mean(0)))
            print(f"  {mode:5s} %below" + "".join(f"{b:16.0%}" for b in below.mean(0)))
        for mode in modes:
            rs = [r for r in rows if r["mode"] == mode]
            print(f"  {mode:5s} true-plan cost mean {np.mean([r['true_cost'] for r in rs]):.2f}")


def plot(rows, modes, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference categorical palette, slots 1-2 (validated: dataviz palette.md); chrome = muted inks
    colors = {"hold": "#2a78d6", "demo": "#eb6834"}
    ink, ink2, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=surface)
    titles = {"raw": "Raw CEM (Gaussian init)", "gmm": "GMM-seeded CEM (it 0 = best of GMM seeds)"}
    for ax, planner in zip(axes, ("raw", "gmm")):
        ax.set_facecolor(surface)
        ends = []
        for mode in modes:
            rs = [r for r in rows if r["mode"] == mode]
            tr = np.array([r[planner] for r in rs])
            true = np.mean([r["true_cost"] for r in rs])
            x = np.arange(tr.shape[1])
            m = tr.mean(0)
            ax.fill_between(x, np.percentile(tr, 25, 0), np.percentile(tr, 75, 0), color=colors[mode], alpha=0.12, lw=0)
            ax.plot(x, m, color=colors[mode], lw=2, label=f"{mode} prefix: best cost found")
            ax.axhline(true, color=colors[mode], lw=1.5, ls=(0, (4, 3)),
                       label=f"{mode} prefix: true plan cost ({true:.1f})")
            ends.append([np.log10(m[-1]), f"{mode}: {m[-1]:.1f}", x[-1]])
        # direct end labels, pushed apart in log space so they never overlap
        ends.sort()
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.13)
        for ly, text, lx in ends:
            ax.annotate(text, (lx, 10**ly), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=ink,
                        bbox=dict(boxstyle="square,pad=0.15", fc=surface, ec="none"))
        if planner == "gmm":
            ax.axvline(10, color=ink2, lw=0.8, ls=":")
            ax.annotate("eval default (10 its)", (10, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                        textcoords="offset points", fontsize=8, color=ink2)
        ax.set_title(titles[planner], loc="left", fontsize=11, color=ink)
        ax.set_xlabel("CEM iteration")
        ax.set_yscale("log")
        ax.grid(True, axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, tr.shape[1] - 1 + 4)
    axes[0].set_ylabel("best world-model cost so far (log)")
    n = len([r for r in rows if r["mode"] == modes[0]])
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(f"Best CEM cost vs the ground-truth plan's cost (mean of {n} held-out episodes, band = IQR)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"wrote {path}")


def plot_planners(rows, mode, path):
    """one seed mode: raw CEM vs GMM-seeded CEM on one axis, with that mode's true-plan cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"raw": "#2a78d6", "gmm": "#eb6834"}  # reference categorical slots 1-2
    names = {"raw": "Raw CEM", "gmm": "GMM-seeded CEM"}
    ink, ink2, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})
    rs = [r for r in rows if r["mode"] == mode]
    true = np.mean([r["true_cost"] for r in rs])

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    ends = []
    for planner in ("raw", "gmm"):
        tr = np.array([r[planner] for r in rs])
        x, m = np.arange(tr.shape[1]), tr.mean(0)
        ax.fill_between(x, np.percentile(tr, 25, 0), np.percentile(tr, 75, 0), color=colors[planner], alpha=0.12, lw=0)
        ax.plot(x, m, color=colors[planner], lw=2, label=f"{names[planner]}: best cost found")
        ends.append([np.log10(m[-1]), f"{names[planner]}: {m[-1]:.1f}", x[-1]])
    ax.axhline(true, color=ink2, lw=1.5, ls=(0, (4, 3)), label=f"ground-truth plan cost ({true:.1f})")
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.13)
    for ly, text, lx in ends:
        ax.annotate(text, (lx, 10**ly), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=ink,
                    bbox=dict(boxstyle="square,pad=0.15", fc=surface, ec="none"))
    ax.set_yscale("log")
    ax.set_xlabel("CEM iteration (GMM: iteration 0 = best of the GMM seeds)")
    ax.set_ylabel("best world-model cost so far (log)")
    ax.grid(True, axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, len(rs[0]["raw"]) - 1 + 8)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Raw-CEM vs GMM-seeded CEM", loc="left", fontsize=12, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"wrote {path}")


def main(c: ConvergenceArgs):
    os.makedirs(c.out_dir, exist_ok=True)
    jpath = os.path.join(c.out_dir, "cem_convergence.json")
    if c.plot_only:
        saved = json.load(open(jpath))
        rows, modes = saved["rows"], tuple(saved["args"]["modes"])
    else:
        rows, modes = run(c), c.modes
        json.dump(dict(args=asdict(c), rows=rows), open(jpath, "w"), indent=1)
    summarize(rows, modes)
    plot(rows, modes, os.path.join(c.out_dir, "cem_convergence.png"))
    for mode in modes:
        plot_planners(rows, mode, os.path.join(c.out_dir, f"cem_vs_gmm_{mode}.png"))


if __name__ == "__main__":
    main(tyro.cli(ConvergenceArgs))
