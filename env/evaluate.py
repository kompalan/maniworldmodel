from collections import defaultdict
from typing import Callable
import numpy as np
import torch
from mani_skill.utils import common
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ..transforms import preprocess_actions, preprocess_pixels
from ..config import Args

def _append_episode_metrics(eval_metrics, episode_metrics, mask=None):
    if episode_metrics is None:
        return 0

    completed = None
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        completed = int(mask.sum())

    for k, v in episode_metrics.items():
        values = np.asarray(v)
        if mask is not None and values.ndim > 0 and values.shape[0] == mask.shape[0]:
            values = values[mask]
        if values.ndim == 0:
            eval_metrics[k].append(values)
        else:
            eval_metrics[k].extend(list(values))

    # Fallback when mask is unavailable (e.g., single-env final_info dict)
    if completed is None:
        first_value = next(iter(episode_metrics.values()), None)
        if first_value is None:
            return 0
        first_value = np.asarray(first_value)
        return int(first_value.shape[0]) if first_value.ndim > 0 else 1
    return completed

def _frame(obs):
    rgb = common.to_numpy(obs["sensor_data"]["base_camera"]["rgb"])
    # preprocess_pixels(torch.tensor(rgb), args.env_args.image_size)

    return rgb
 
def _base_env(eval_envs):
    """the (first) unwrapped ManiSkill env behind a SyncVectorEnv or ManiSkillVectorEnv."""
    envs = getattr(eval_envs, "envs", None)
    return (envs[0] if envs is not None else eval_envs).unwrapped


def hold_action(eval_envs) -> torch.Tensor:
    """the raw action (A,) that keeps the robot where it is: every controller's current
    joint target, mapped back through its action normalization.

    Only absolute joint-position controllers (pd_joint_pos) are supported; for delta
    controllers the no-motion action is simply zeros. Call it right after reset, when the
    targets equal the reset qpos.
    """
    parts = []
    for name, c in _base_env(eval_envs).agent.controller.controllers.items():
        cfg = c.config
        assert not getattr(cfg, "use_delta", True), f"controller '{name}' is not absolute pd_joint_pos"
        target = c._target_qpos[0]  # (n_joints,) of env 0
        idx = getattr(c, "mimic_control_joint_indices", None)
        if idx is not None and len(idx):  # mimic: one action per control joint
            target = target[torch.as_tensor(idx).flatten()]
        if c._normalize_action:  # inverse of clip_and_scale_action: [low, high] -> [-1, 1]
            low = torch.as_tensor(c._original_single_action_space.low, device=target.device)
            high = torch.as_tensor(c._original_single_action_space.high, device=target.device)
            target = (2 * (target - low) / (high - low) - 1).clamp(-1, 1)
        parts.append(target.float().cpu())
    act = torch.cat(parts)
    assert act.shape == eval_envs.single_action_space.shape, (act.shape, eval_envs.single_action_space.shape)
    return act


def _step(eval_envs, act):
    """Step the vector env with one raw action of shape (A,) or (num_envs, A)."""
    act = np.asarray(common.to_numpy(act), dtype=np.float32)
    if act.ndim == 1:
        act = act[None]
    return eval_envs.step(act)


def evaluate(
    n: int,
    seed: int,
    sample_fn: Callable,
    eval_envs: ManiSkillVectorEnv,
    seed_actions: torch.Tensor | np.ndarray | None = None,
    seed_obs: torch.Tensor | np.ndarray | None = None,
    hold_steps: int = 0,
    hold_gripper: float | None = None,
):
    """
    Evaluate the agent on the evaluation environments for at least n episodes.

    Args:
        n: The minimum number of episodes to evaluate.
        seed: The seed passed to eval_envs.reset (use the demo's episode_seed to replay it).
        sample_fn: Called as sample_fn(obs, actions_so_far) and must return the raw actions
            to execute, shape (k, A). obs is the env's frame stack (num_envs, T, H, W, 3),
            frames spaced `frameskip` env steps apart with the current frame last;
            actions_so_far is every raw action executed so far, shape (N, A).
        eval_envs: The evaluation environments.
        seed_actions: Optional raw (unnormalized) ground-truth actions, shape (N, A), executed
            right after reset before the agent acts. They also start actions_so_far.
        seed_obs: Optional ground-truth frames (T, H, W, 3) to hand to the first sample_fn call
            in place of the rendered frame stack (same T as the stack). The mean pixel error
            between the two is printed so replay drift is visible.
        hold_steps: Instead of a ground-truth prefix, execute this many copies of the no-motion
            action (see hold_action) right after reset; they start actions_so_far. No demo data
            is used. Mutually exclusive with seed_actions / seed_obs.
        hold_gripper: If given, overrides the last (gripper) dimension of the hold action, e.g.
            -1 (closed) to match demos whose gripper action is always -1. Holding the reset
            gripper opening (+1) can be far outside the training action distribution.

    Returns:
        A dictionary containing the evaluation results.
    """
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset(seed=seed)
        action_dim = eval_envs.single_action_space.shape[0]
        actions_so_far = torch.empty(0, action_dim)

        if hold_steps > 0:
            assert seed_actions is None and seed_obs is None, "hold_steps replaces the ground-truth seed"
            hold = hold_action(eval_envs)
            if hold_gripper is not None:
                hold[-1] = hold_gripper
            seed_actions = hold.repeat(hold_steps, 1)  # executed below like a GT prefix

        # seed the episode with the ground-truth (or hold) prefix
        if seed_actions is not None:
            seed_actions = torch.as_tensor(seed_actions, dtype=torch.float32)
            for k, act in enumerate(seed_actions):
                obs, _, _, truncated, info = _step(eval_envs, act)
                # check every step: the vector env autoresets on truncation, so the remaining
                # seed actions would silently replay in a fresh (differently laid out) episode
                assert not np.asarray(truncated).any(), (
                    f"episode ended at seed step {k + 1}/{len(seed_actions)}; raise max_episode_steps"
                )
            actions_so_far = seed_actions.clone()
        obs = torch.as_tensor(obs)
        if seed_obs is not None:
            seed_obs = torch.as_tensor(seed_obs)
            assert seed_obs.shape == obs.shape[1:], (
                f"seed_obs {tuple(seed_obs.shape)} must match the frame stack {tuple(obs.shape[1:])}"
            )
            err = (obs.float() - seed_obs.float()).abs().mean(dim=(0, 2, 3, 4))
            print(f"seed replay: mean |rendered - ground truth| per frame = {err.tolist()}")
            obs = seed_obs.unsqueeze(0).expand_as(obs).clone()

        eps_count = 0
        while eps_count < n:
            actions = torch.as_tensor(sample_fn(obs, actions_so_far), dtype=torch.float32).reshape(-1, action_dim)
            obs, _, _, truncated, info = None, None, None, None, None
            for j, act in enumerate(actions):
                obs, _, _, truncated, info = _step(eval_envs, act)
                # the episode can end mid-plan: stop there, otherwise the vector env autoresets
                # and the rest of the plan runs (uncounted) in a new episode
                if np.asarray(truncated).any():
                    break
            actions_so_far = torch.cat([actions_so_far, actions[: j + 1]])
            obs = torch.as_tensor(obs)

            if not np.asarray(truncated).any():
                continue

            # Gymnasium vector envs may expose episode summaries in either:
            # 1) info["final_info"] (plus optional info["_final_info"] mask), or
            # 2) info["episode"] (plus optional info["_episode"] mask).
            if "final_info" in info:
                final_info = info["final_info"]
                if isinstance(final_info, dict):
                    mask = info.get("_final_info")
                    episode = final_info.get("episode")
                    eps_count += _append_episode_metrics(eval_metrics, episode, mask=mask)
                else:
                    # Usually an array/list of per-env dicts (None for unfinished envs).
                    completed = 0
                    for one_final_info in final_info:
                        if not one_final_info or "episode" not in one_final_info:
                            continue
                        completed += _append_episode_metrics(
                            eval_metrics, one_final_info["episode"]
                        )
                    eps_count += completed
            elif "episode" in info:
                eps_count += _append_episode_metrics(
                    eval_metrics, info["episode"], mask=info.get("_episode")
                )
    
    for k in eval_metrics.keys():
        eval_metrics[k] = np.asarray(eval_metrics[k])
    return eval_metrics
