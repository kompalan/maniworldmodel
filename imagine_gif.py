"""
GIFs of an imagined (world-model) rollout next to the real one.

For each held-out episode: set up the planner's first call like eval.py (demo or hold prefix), make
the first plan (before any step is taken), and decode the world model's imagined rollout of it.
Then run the real rollout in the simulator:
  --plan gmm, --real-rollout replan     the planner as in eval.py: replan every chunk (receding horizon 1)
  --plan gmm, --real-rollout open-loop  execute the first plan unchanged
  --plan demo                           replay the demo's remaining actions
Each GIF frame is four panels, one frame per action chunk:
  reconstructed imagined rollout (first plan) | reconstructed goal latent | real rollout | real goal
The decoder must be trained on the same LeWM checkpoint's latents (lewm.latent_reconstruction).

Run from examples/baselines:
    lewm/.venv/bin/python -m lewm.imagine_gif --ckpt-path lewm/runs/lewm_bn_ep8/latest.pt \
        --decoder-ckpt lewm/runs/latent_reconstructor_ep8/latest.pt --plan demo --seed-mode demo
"""
import os
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw

from .config import Args, EvalArgs
from .dataset import ManiSkillTrajectoryDataset
from .env.evaluate import _base_env, _step, hold_action
from .env.make_env import make_eval_envs
from .eval import _plan_context, cem_plan, cem_plan_with_gmm_prior
from .gmm_prior import GMMPrior
from .latent_reconstruction import LatentReconstructor, target_pixels
from .transforms import preprocess_pixels
from .utils import _dataclass_from_dict, get_model


@dataclass
class GifArgs:
    ckpt_path: str = "lewm/runs/lewm_bn_ep8/latest.pt"
    decoder_ckpt: str = "lewm/runs/latent_reconstructor_ep8/latest.pt"
    """LatentReconstructor trained on ckpt_path's encoder"""
    heldout_dataset: str = EvalArgs.heldout_dataset
    out_dir: str = "runs/imagine_gifs"
    episodes: tuple[int, ...] = (0, 1, 2, 3)
    seed_mode: str = "demo"
    """"demo": first H ground-truth chunks before the plan; "hold": H chunks of no-motion action"""
    plan: str = "demo"
    """"demo": the demo's own next actions; "gmm": the GMM-seeded CEM plan; "cem": raw CEM, no prior
    (both with eval.py defaults)"""
    real_rollout: str = "replan"
    """for --plan gmm/cem: "replan" runs the planner closed-loop like eval.py; "open-loop" executes the first plan"""
    max_episode_steps: int = EvalArgs.max_episode_steps
    plan_horizon: int = EvalArgs.plan_horizon
    gmm_prior_ckpt: str = "runs/gmm_prior/gmm_prior.pt"
    hold_gripper: float | None = EvalArgs.hold_gripper
    fps: int = 3
    device: str = "cuda"


def header(img: np.ndarray, text: str) -> np.ndarray:
    """(h, w, 3) uint8 with a 20 px label bar on top."""
    bar = Image.new("RGB", (img.shape[1], 20), (252, 252, 251))
    ImageDraw.Draw(bar).text((4, 4), text, fill=(31, 31, 30))
    return np.concatenate([np.asarray(bar), img], 0)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(3, S, S) float [0, 1] -> (S, S, 3) uint8."""
    return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def main(g: GifArgs):
    assert g.seed_mode in ("demo", "hold") and g.plan in ("demo", "gmm", "cem")
    assert g.real_rollout in ("replan", "open-loop"), g.real_rollout
    ckpt = torch.load(g.ckpt_path, map_location="cpu", weights_only=False)
    args = _dataclass_from_dict(Args, ckpt["args"])
    H, fs, S, T = args.model_args.history_size, args.env_args.frameskip, args.env_args.image_size, g.plan_horizon
    dev = g.device

    data = ManiSkillTrajectoryDataset(g.heldout_dataset, num_steps=H + 1, frameskip=fs, load_count=max(g.episodes) + 1)
    A = data.action_dim
    model = get_model(args, A)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval().requires_grad_(False)
    am = torch.as_tensor(ckpt["action_mean"], dtype=torch.float32)
    sd = torch.as_tensor(ckpt["action_std"], dtype=torch.float32)

    dck = torch.load(g.decoder_ckpt, map_location="cpu", weights_only=False)
    ra = dck["recon_args"]
    if os.path.abspath(ra["ckpt_path"]) != os.path.abspath(g.ckpt_path):
        print(f"WARNING: decoder was trained on {ra['ckpt_path']}, not {g.ckpt_path}")
    decoder = LatentReconstructor(args.model_args.embedding_dim, ra["n_heads"], ra["n_layers"], S, ra["patch_size"])
    decoder.load_state_dict(dck["model"])
    decoder.to(dev).eval()

    prior = GMMPrior.from_checkpoint(g.gmm_prior_ckpt, device=dev) if g.plan == "gmm" else None
    first_args = EvalArgs(exp_name="gif", ckpt_path=g.ckpt_path, plan_receding_horizon=T, plan_horizon=T)
    replan_args = EvalArgs(exp_name="gif", ckpt_path=g.ckpt_path, plan_receding_horizon=1, plan_horizon=T)

    envs = make_eval_envs(
        env_id=args.env_args.env_id, num_envs=1, sim_backend="physx_cpu",
        env_kwargs={**data.env_info["env_kwargs"], "max_episode_steps": g.max_episode_steps},
        other_kwargs={"obs_horizon": H + 1, "frameskip": fs},
    )
    encode = lambda frames: model.encode(dict(pixels=preprocess_pixels(frames[None], S).to(dev)))["emb"][0]
    os.makedirs(g.out_dir, exist_ok=True)

    for i in g.episodes:
        ep = data.episodes[i]
        o0, a0, _ = data.episode_windows[i]
        goal = torch.tensor(data.obs[o0 + ep["elapsed_steps"]])
        demo = torch.tensor(data.actions[a0 : a0 + ep["elapsed_steps"]])

        obs, _ = envs.reset(seed=ep["episode_seed"])
        if g.seed_mode == "hold":
            act = hold_action(envs)
            if g.hold_gripper is not None:
                act[-1] = g.hold_gripper
            executed, start = act.repeat(H * fs, 1), 0
        else:
            executed, start = demo[: H * fs].float(), H * fs
        for a in executed:
            obs, *_ = _step(envs, a)
        obs = torch.as_tensor(obs)
        pixels, goal_px, hist = _plan_context(args, goal, am, sd, obs, executed, None, dev)

        gen = torch.Generator(device=dev).manual_seed(i)

        def planner(eval_args, obs, actions_so_far):
            if g.plan == "gmm":
                return cem_plan_with_gmm_prior(args, eval_args, model, prior, goal, am, sd, obs, actions_so_far, generator=gen)
            return cem_plan(args, eval_args, model, goal, am, sd, obs, actions_so_far, generator=gen)

        if g.plan == "demo":
            d = demo[start : start + T * fs]
            raw = torch.cat([d, demo[-1:].expand(T * fs - len(d), -1)])
        else:
            raw = planner(first_args, obs, executed)
        plan = ((raw - am) / sd).reshape(1, 1, T, fs * A).to(dev)

        # imagined rollout of the first plan: predicted embeddings of o_{t+1..t+T}
        cand = torch.cat([hist, plan], dim=2)
        pred = model.rollout(dict(pixels=pixels, action=cand), cand)["predicted_emb"][0, 0, H:]  # (T, D)
        imagined = decoder(torch.cat([encode(obs[0, -1:]), pred]))  # step 0 = the observed o_t

        # real rollout, one frame per chunk until the env's step budget runs out
        budget = (g.max_episode_steps - executed.shape[0]) // fs
        if g.plan == "demo":
            rest = demo[start:]
            chunks = [rest[k * fs : (k + 1) * fs] for k in range(min(budget, -(-len(rest) // fs)))]
        elif g.real_rollout == "open-loop":
            chunks = [raw[k * fs : (k + 1) * fs] for k in range(min(budget, T))]
        else:
            chunks = None  # replan below
        real, succ, actions_so_far = [obs[0, -1]], [False], executed
        for k in range(budget if chunks is None else len(chunks)):
            if chunks is None:
                chunk = raw[:fs] if k == 0 else planner(replan_args, obs, actions_so_far)
            else:
                chunk = chunks[k]
            for a in chunk:
                o, *_ = _step(envs, a)
            actions_so_far = torch.cat([actions_so_far, chunk.float()])
            obs = torch.as_tensor(o)
            real.append(obs[0, -1])
            succ.append(bool(_base_env(envs).evaluate()["success"][0]))
        real_px = target_pixels(torch.stack(real).to(dev), S)

        goal_dec = to_uint8(decoder(encode(goal[None]))[0])
        goal_img = to_uint8(target_pixels(goal[None].to(dev), S)[0])
        real_name = {"demo": "demo replay", "open-loop": "first plan, open loop", "replan": "planner, replanning"}[
            "demo" if g.plan == "demo" else g.real_rollout]

        frames = []
        for k in range(max(T + 1, len(real))):
            ki, kr = min(k, T), min(k, len(real) - 1)
            itag = f"imagined t+{ki}" if k <= T else f"imagined t+{T} (plan end)"
            rtag = f"real t+{kr}" + ("  SUCCESS" if succ[kr] else "")
            frames.append(np.concatenate([
                header(to_uint8(imagined[ki]), itag),
                header(goal_dec, "reconstructed goal"),
                header(to_uint8(real_px[kr]), f"{rtag}"),
                header(goal_img, "real goal"),
            ], 1))
        frames += [frames[-1]] * g.fps  # hold the last frame
        name = g.plan if g.plan == "demo" else f"{g.plan}_{g.real_rollout}"
        path = os.path.join(g.out_dir, f"ep{i}_{g.seed_mode}_{name}.gif")
        imageio.mimsave(path, frames, duration=1000 / g.fps, loop=0)
        print(f"wrote {path}  ({real_name}, {len(real) - 1} chunks, success={any(succ)})")
    envs.close()


if __name__ == "__main__":
    main(tyro.cli(GifArgs))
