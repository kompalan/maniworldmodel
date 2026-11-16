from .env.evaluate import evaluate
from .env.make_env import make_eval_envs
from .utils import _dataclass_from_dict, get_model, Logger

import enum
import pickle
import types
from dataclasses import dataclass, asdict

from functools import partial

from .transforms import preprocess_actions, preprocess_pixels
from . import config
from .config import Args, EvalArgs
from .jepa import JEPA
from .dataset import ManiSkillTrajectoryDataset
from .action_prior import ActionPrior
from .gmm_prior import GMMPrior

from collections import defaultdict

from mani_skill.utils.visualization.misc import images_to_video

import numpy as np
import torch
import tyro
import wandb
from pathlib import Path

class _CkptUnpickler(pickle.Unpickler):
    """Resolve classes pickled under `__main__` (train.py run as a script) to lewm.config."""

    def find_class(self, module, name):
        if module == "__main__" and hasattr(config, name):
            return getattr(config, name)
        return super().find_class(module, name)


# torch.load's pickle_module only needs `Unpickler` and `load`
_ckpt_pickle = types.SimpleNamespace(
    __name__="_ckpt_pickle",
    Unpickler=_CkptUnpickler,
    load=lambda f, **kw: _CkptUnpickler(f, **kw).load(),
)

def _plan_context(args: Args, goal, action_mean, action_std, obs, actions_so_far, frames, device):
    """preprocess the planner inputs shared by every CEM variant.

    Returns pixels (1, 1, H, 3, S, S) of the latest H frames [o_{t-H+1}, ..., o_t], goal pixels
    (1, 1, 1, 3, S, S) and the normalized executed chunks (1, 1, H-1, fs*A) taken from the first
    H-1 of those frames. The first planned chunk is paired with the real current frame o_t, so
    the world model never has to re-predict an observation it was given.
    """
    H = args.model_args.history_size
    fs = args.env_args.frameskip
    A = actions_so_far.shape[-1]
    if frames is not None:
        new = obs[0] if not frames else obs[0, -1:]
        frames.extend(torch.cat([f, goal], dim=1).cpu().numpy() for f in new)
    assert actions_so_far.shape[0] >= (H - 1) * fs, (
        f"need {(H - 1) * fs} executed actions of history, got {actions_so_far.shape[0]} (seed the episode)"
    )
    pixels = preprocess_pixels(obs[:, -H:], args.env_args.image_size).unsqueeze(1).to(device)
    goal_px = preprocess_pixels(goal[None, None], args.env_args.image_size).unsqueeze(1).to(device)
    hist = actions_so_far[actions_so_far.shape[0] - (H - 1) * fs :]  # (H-1)*fs raw actions (may be empty)
    hist = (hist - action_mean) / action_std
    hist = preprocess_actions(hist, 0.0).reshape(1, 1, H - 1, fs * A).to(device)
    return pixels, goal_px, hist


def _score(world_model: JEPA, pixels, goal_px, hist, future: torch.Tensor, chunk: int) -> torch.Tensor:
    """world-model cost of each future (N, T, fs*A) after the history, scored `chunk` at a time."""
    costs = []
    for f in future.split(chunk):
        # [a_{t-H+1}, ..., a_{t-1} | a_t, ..., a_{t+T-1}]: rollout pairs the first H with the H frames
        candidates = torch.cat([hist.expand(1, len(f), -1, -1), f[None]], dim=2)  # (1, n, H-1+T, fs*A)
        info = dict(pixels=pixels, goal=goal_px, action=candidates)
        costs.append(world_model.get_cost(info, candidates)[0])
    return torch.cat(costs)


def _cem(world_model: JEPA, eval_args: EvalArgs, pixels, goal_px, hist, mean, std, iters: int, generator=None):
    """standard CEM over normalized future chunks from N(mean, std^2); returns the final mean.
    generator: torch.Generator on mean's device for reproducible sampling (None = global RNG)."""
    S, K = eval_args.cem_num_samples, eval_args.cem_topk
    cost = None
    for _ in range(iters):
        future = mean + std * torch.randn(S, *mean.shape, device=mean.device, generator=generator)
        future[0] = mean  # always score the current mean
        cost = _score(world_model, pixels, goal_px, hist, future, S)
        elite = future[cost.topk(K, largest=False).indices]  # (K, T, fs*A)
        mean, std = elite.mean(0), elite.std(0)
    if cost is not None:
        print(f"Final Cost: {cost.min()}")
    return mean


def _to_env_actions(mean, eval_args: EvalArgs, action_mean, action_std):
    A = action_mean.shape[-1]
    plan = mean[: eval_args.plan_receding_horizon].reshape(-1, A).cpu()
    return plan * action_std + action_mean


@torch.no_grad()
def cem_plan_with_action_prior(
    args: Args,
    eval_args: EvalArgs,
    world_model: JEPA,
    prior: ActionPrior,
    goal: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    obs: torch.Tensor,
    actions_so_far: torch.Tensor,
    frames: list | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """CEM seeded and pruned with a DDIM action prior (see lewm.action_prior).

    1. draw `prior_num_samples` diverse plans on the action manifold: the prior continues the
       executed actions chunk by chunk (inpainting), so plans start from the current state
    2. score them with the world model and keep the best `cem_topk` as elites
    3. fit the CEM Gaussian to those elites and run `prior_cem_iters` rounds of standard CEM

    Same arguments and return value as cem_plan, plus the prior.
    """
    device = next(world_model.parameters()).device
    pixels, goal_px, hist = _plan_context(args, goal, action_mean, action_std, obs, actions_so_far, frames, device)
    T, K = eval_args.plan_horizon, eval_args.cem_topk
    A = actions_so_far.shape[-1]
    am, sd = action_mean.to(device), action_std.to(device)

    # the prior has its own normalization: go through raw actions on the way in and out
    # whole executed chunks up to o_t (the prior uses the last horizon-1 of them)
    n_exec = actions_so_far.shape[0] // args.env_args.frameskip * args.env_args.frameskip
    context = prior.normalize(actions_so_far[actions_so_far.shape[0] - n_exec :].to(device).float())  # (C, fs*A)
    seed = prior.sample_continuation(
        context, T, eval_args.prior_num_samples, eval_args.prior_inference_steps, eval_args.prior_eta, generator
    )  # (N, T, fs*A), prior units
    seed = (prior.unnormalize(seed) - am) / sd  # raw (N, T*fs, A) -> world-model units
    seed = seed.reshape(-1, T, args.env_args.frameskip * A)

    cost = _score(world_model, pixels, goal_px, hist, seed, eval_args.cem_num_samples)
    elite = seed[cost.topk(K, largest=False).indices]
    print(f"Prior seed cost: best {cost.min():.4f}, elite mean {cost.topk(K, largest=False).values.mean():.4f}")

    mean = _cem(
        world_model, eval_args, pixels, goal_px, hist, elite.mean(0), elite.std(0), eval_args.prior_cem_iters, generator
    )
    return _to_env_actions(mean, eval_args, action_mean, action_std)


@torch.no_grad()
def cem_plan_with_gmm_prior(
    args: Args,
    eval_args: EvalArgs,
    world_model: JEPA,
    prior: GMMPrior,
    goal: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    obs: torch.Tensor,
    actions_so_far: torch.Tensor,
    frames: list | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """CEM seeded with candidates from a GMM fit to the training actions (see lewm.gmm_prior).

    1. draw `prior_num_samples` plans from the GMM, conditioned on the executed action chunks
       (each component's Gaussian conditional), so candidates start from the current state
    2. score them with the world model and keep the best `cem_topk` as elites
    3. fit the CEM Gaussian to those elites and run `prior_cem_iters` rounds of standard CEM

    Same arguments and return value as cem_plan, plus the prior.
    """
    device = next(world_model.parameters()).device
    pixels, goal_px, hist = _plan_context(args, goal, action_mean, action_std, obs, actions_so_far, frames, device)
    T, K, fs = eval_args.plan_horizon, eval_args.cem_topk, args.env_args.frameskip
    A = actions_so_far.shape[-1]
    am, sd = action_mean.to(device), action_std.to(device)

    # the GMM has its own normalization: go through raw actions on the way in and out
    n_exec = actions_so_far.shape[0] // fs * fs  # whole executed chunks up to o_t
    context = prior.normalize(actions_so_far[actions_so_far.shape[0] - n_exec :].to(prior.device).float())
    seed = prior.sample_continuation(context, T, eval_args.prior_num_samples, generator)  # (N, T, fs*A)
    seed = (prior.unnormalize(seed).to(device) - am) / sd  # raw (N, T*fs, A) -> world-model units
    seed = seed.reshape(-1, T, fs * A)

    cost = _score(world_model, pixels, goal_px, hist, seed, eval_args.cem_num_samples)
    top = cost.topk(K, largest=False)
    elite = seed[top.indices]
    print(f"GMM seed cost: best {cost.min():.4f}, elite mean {top.values.mean():.4f}")

    mean = _cem(
        world_model, eval_args, pixels, goal_px, hist, elite.mean(0), elite.std(0), eval_args.prior_cem_iters, generator
    )
    return _to_env_actions(mean, eval_args, action_mean, action_std)


@torch.no_grad()
def cem_plan(
    args: Args,
    eval_args: EvalArgs,
    world_model: JEPA,
    goal: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    obs: torch.Tensor,
    actions_so_far: torch.Tensor,
    frames: list | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Plan with the cross-entropy method in the world model's latent space.

    Args:
        goal: goal frame (h, w, 3) uint8
        action_mean, action_std: per-step action normalization used in training, shape (A,)
        obs: env frame stack (1, >=H, h, w, 3) uint8 ending at the current frame o_t, frames
            `frameskip` env steps apart (see make_eval_envs). The latest H are the context.
        actions_so_far: every raw action executed so far, (N, A). The last H-1 chunks are the
            ones taken from the context frames before o_t.
        frames: if given, the observations the planner sees are appended to it as
            [observation | goal] side-by-side uint8 frames (h, 2w, 3). The first call adds the
            whole seeded stack, later calls add only the current frame o_t.
        generator: torch.Generator on the model's device for reproducible CEM noise

    Returns:
        raw actions (receding_horizon * frameskip, A) to execute from o_t
    """
    fs = args.env_args.frameskip
    A = actions_so_far.shape[-1]
    device = next(world_model.parameters()).device
    pixels, goal_px, hist = _plan_context(args, goal, action_mean, action_std, obs, actions_so_far, frames, device)

    # Gaussian over the normalized future action chunks
    T = eval_args.plan_horizon
    # start from the last executed raw action held for the whole horizon (for pd_joint_pos
    # that's "keep the current joint targets"), in normalized units
    last = (actions_so_far[-1] - action_mean) / action_std
    mean = last.to(device).repeat(T, fs)  # (T, fs*A)
    std = torch.full((T, fs * A), eval_args.cem_var_scale, device=device)
    mean = _cem(world_model, eval_args, pixels, goal_px, hist, mean, std, eval_args.cem_iters, generator)
    return _to_env_actions(mean, eval_args, action_mean, action_std)


@torch.no_grad()
def sanity_check(
    args: Args, 
    world_model: JEPA,
    goal: torch.Tensor, 
    ground_truth_action: torch.Tensor,
    obs: torch.Tensor
):
    goal = goal.expand(1, 1, -1, -1, -1)
    obs = obs.unsqueeze(0)
    obs = preprocess_pixels(obs, args.env_args.image_size).unsqueeze(0)
    goal = preprocess_pixels(goal, args.env_args.image_size).unsqueeze(0)
    action = preprocess_actions(ground_truth_action, 0.0).reshape(1, 1, -1, args.env_args.frameskip * ground_truth_action.shape[-1])

    info_dict = dict(
        pixels=obs,
        action=action,
        goal=goal
    )

    ground_truth_cost = world_model.get_cost(info_dict, action)

    random_actions = torch.normal(torch.zeros_like(action), 1.0)
    random_actions = random_actions.clip(-1.0, 1.0)

    print(random_actions.shape)
    info_dict = dict(
        pixels=obs,
        action=random_actions,
        goal=goal
    )
    random_cost = world_model.get_cost(info_dict, random_actions)

    return {"gt_cost": ground_truth_cost, "random_cost": random_cost}

def load_checkpoint(ckpt_path: str, device: str | None = None):
    """Rebuild the training Args and the JEPA world model from a train.py checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=_ckpt_pickle)
    args = _dataclass_from_dict(Args, ckpt["args"])
    if device is not None:
        args.device = device

    # train.py saves the per-step action normalization (shape (action_dim,))
    action_dim = len(ckpt["action_mean"])
    model = get_model(args, action_dim)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    model.requires_grad_(False)

    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']}, step {ckpt['global_step']}, action_dim={action_dim})")
    return args, model, ckpt

@torch.no_grad()
def eval(
    args: Args,
    eval_args: EvalArgs,
    world_model: JEPA,
    dataset: torch.utils.data.DataLoader | ManiSkillTrajectoryDataset,
    prior: ActionPrior | None = None,
    gmm_prior: GMMPrior | None = None,
    logger: Logger | None = None,
):
    envs = make_eval_envs(
        env_id=args.env_args.env_id,
        num_envs=1,
        sim_backend="physx_cpu",
        env_kwargs={
            **dataset.env_info["env_kwargs"],
            "max_episode_steps": eval_args.max_episode_steps,
        },
        other_kwargs={
            # history_size frames the seed actions were taken from + the current frame
            "obs_horizon": args.model_args.history_size + 1,
            "frameskip": args.env_args.frameskip,
        },
        video_dir=eval_args.video_dir,
    )

    H, fs = args.model_args.history_size, args.env_args.frameskip
    
    assert eval_args.seed_mode in ("demo", "hold"), f"unknown seed_mode {eval_args.seed_mode!r}"
    hold = eval_args.seed_mode == "hold"
    seed_steps = H * fs  # demo or hold prefix, both executed in the env
    min_steps = seed_steps + eval_args.plan_receding_horizon * fs  # seed prefix + one executed plan
    assert eval_args.max_episode_steps >= min_steps, (
        f"max_episode_steps={eval_args.max_episode_steps} < {min_steps} (seed {seed_steps} + one plan step)"
    )
    action_mean = torch.as_tensor(dataset.action_mean, dtype=torch.float32)
    action_std = torch.as_tensor(dataset.action_std, dtype=torch.float32)
    results = defaultdict(list)
    for i, ep_data in enumerate(dataset.episodes[:eval_args.num_episodes]):
        obs_start, act_start, _ = dataset.episode_windows[i]
        # goal = the demo's final frame (episodes store elapsed_steps + 1 observations)
        goal = torch.tensor(dataset.obs[obs_start + ep_data["elapsed_steps"]])

        # # seed the rollout with the first H ground-truth action chunks and the frames they were
        # # taken from: the env replays them, so planning starts from the GT state o_{H*fs} with
        # # history (o_0, o_fs, ..., o_{H*fs}) and (a_0, ..., a_{H*fs - 1})
        # prefix = ep_data["elapsed_steps"] - 25
        seed_actions = torch.tensor(dataset.actions[act_start:act_start + H * fs])  # raw (H*fs, A)
        seed_obs = torch.tensor(dataset.obs[obs_start:obs_start + H * fs + 1:fs])  # (H+1, h, w, 3)

        # fast-forward: replay the demo up to `steps_before_goal` env steps before its end
        prefix = eval_args.play_prefix          # env steps replayed before planning
        print("Steps left in episode:", ep_data["elapsed_steps"] - prefix)
        # assert prefix >= H * fs, f"prefix {prefix} too short for a {H + 1}-frame stack"
        # seed_actions = torch.tensor(dataset.actions[act_start : act_start + prefix])            # raw (prefix, A)
        # seed_obs = torch.tensor(dataset.obs[obs_start + prefix - H * fs : obs_start + prefix + 1 : fs])  # (H+1, ...) ending at o_prefix

        frames = [] if eval_args.video_dir else None
        # planner noise seeded per episode: reruns with the same --seed plan identically
        gen = torch.Generator(device=next(world_model.parameters()).device).manual_seed(eval_args.seed * 10_000 + i)
        if gmm_prior is not None:
            plan_fn = partial(
                cem_plan_with_gmm_prior, args, eval_args, world_model, gmm_prior, goal, action_mean, action_std,
                frames=frames, generator=gen,
            )
        elif prior is None:
            plan_fn = partial(
                cem_plan, args, eval_args, world_model, goal, action_mean, action_std, frames=frames, generator=gen
            )
        else:
            plan_fn = partial(
                cem_plan_with_action_prior, args, eval_args, world_model, prior, goal, action_mean, action_std,
                frames=frames, generator=gen,
            )
        if hold:
            # H*fs hold actions: same history length as the demo seed (the priors read whole chunks)
            seed_kw = dict(hold_steps=H * fs, hold_gripper=eval_args.hold_gripper)
        else:
            seed_kw = dict(seed_actions=seed_actions, seed_obs=seed_obs)
        metrics = evaluate(1, ep_data["episode_seed"], plan_fn, envs, **seed_kw)
        if frames:
            images_to_video(frames, eval_args.video_dir, f"planner_obs_{i}", fps=5, verbose=False)
        print(f"episode {i} (seed {ep_data['episode_seed']}): {dict(metrics)}")
        for k, v in metrics.items():
            results[k].extend(np.atleast_1d(v).tolist())

        if logger:
            env_cam = Path(f"{eval_args.video_dir}/{i}.mp4")
            model_view = Path(f"{eval_args.video_dir}/planner_obs_{i}.mp4")

            metrics = { f"eval/{k}": float(v[0]) for k, v in metrics.items() if not k.startswith("_") }

            if env_cam.resolve().exists():
                metrics |= { "videos/environment_cam": wandb.Video(f"{eval_args.video_dir}/{i}.mp4") }

            if model_view.resolve().exists():
                metrics |= { "videos/what_the_model_sees": wandb.Video(f"{eval_args.video_dir}/planner_obs_{i}.mp4") }
            
            logger.log(
                metrics=metrics,
                step=i
            )



    envs.close()
    summary = {k: float(np.mean(v)) for k, v in results.items()}
    print(summary)
    return summary


if __name__ == "__main__":
    eval_args: EvalArgs = tyro.cli(EvalArgs)

    ckpt = torch.load(eval_args.ckpt_path, map_location="cpu", weights_only=False)
    args = _dataclass_from_dict(Args, ckpt["args"])


    dataset = ManiSkillTrajectoryDataset(
        eval_args.heldout_dataset,
        num_steps=args.model_args.history_size + args.model_args.predictor.num_preds,
        frameskip=args.env_args.frameskip,
        load_count=eval_args.num_episodes
    )

    # loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, pin_memory=True, num_workers=4)

    dataset.action_mean = ckpt["action_mean"]
    dataset.action_std = ckpt["action_std"]

    model = get_model(args, dataset.action_dim)
    model.load_state_dict(ckpt["model"])
    device = eval_args.device or (args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.requires_grad_(False)

    prior = None
    if eval_args.action_prior_ckpt is not None:
        prior = ActionPrior.from_checkpoint(eval_args.action_prior_ckpt, device=device)
        assert prior.frameskip == args.env_args.frameskip, "prior and world model frameskip differ"

    gmm_prior = None
    if eval_args.gmm_prior_ckpt is not None:
        assert prior is None, "pass only one of --action-prior-ckpt / --gmm-prior-ckpt"
        gmm_prior = GMMPrior.from_checkpoint(eval_args.gmm_prior_ckpt, device=device)
        assert gmm_prior.frameskip == args.env_args.frameskip, "GMM prior and world model frameskip differ"

    logger = Logger(
        eval_args.wandb_config.wandb_project_name,
        eval_args.wandb_config.wandb_team,
        eval_args.exp_name,
        asdict(eval_args),
        eval_args.wandb_config.enable
    )

    # examples/baselines/lewm/runs/lewm_bn_200ep/latest.pt
    logger.log_artifact(name="lewm_checkpoint", type="model", strpath="lewm/runs/lewm_bn_200ep/latest.pt")

    # examples/baselines/runs/gmm_prior
    logger.log_artifact(name="gmm_checkpoint", type="model", strpath="runs/gmm_prior/gmm_prior.pt")
    
    metrics = eval(args, eval_args, model, dataset, prior, gmm_prior, logger)