"""
ManiSkill trajectory dataset for LeWorldModel training.

Adapted from https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html.
Each item is one episode: a set of `num_windows` sub-trajectories ("windows") sampled
from that episode. A window holds `num_steps` frames spaced `frameskip` env steps apart,
and the `frameskip` raw actions between two frames are concatenated into one action.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional

from mani_skill.utils.io_utils import load_json


class ManiSkillTrajectoryDataset(Dataset):
    """
    Args:
        dataset_file: path to the .h5 file (with a matching .json next to it)
        num_steps: frames per window (history_size + num_preds)
        frameskip: env steps between consecutive frames of a window
        num_windows: windows sampled from an episode per item
        load_count: number of trajectories to load. -1 or None loads all
        success_only: skip unsuccessful trajectories
        camera: which camera's rgb to load
        load_pixels: load camera frames. False keeps only actions (items have no "pixels")
    """

    def __init__(
        self,
        dataset_file: str,
        num_steps: int = 4,
        frameskip: int = 5,
        num_windows: int = 64,
        load_count: Optional[int]=None,
        success_only: bool = False,
        camera: str = "base_camera",
        load_pixels: bool = True,
    ) -> None:
        self.load_pixels = load_pixels
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.num_windows = num_windows

        json_data = load_json(dataset_file.replace(".h5", ".json"))
        self.env_info = json_data["env_info"]
        self.env_id = self.env_info["env_id"]

        self.episodes = json_data["episodes"]
        if load_count is None:
            load_count = len(self.episodes)
        self.episodes = self.episodes[:load_count]
        if success_only:
            assert all("success" in e for e in self.episodes), "episodes have no success attribute"
            self.episodes = [e for e in self.episodes if e["success"]]

        data = h5py.File(dataset_file, "r")

        # preallocate so peak memory is ~1x the frames (PushCube front48: ~14GB uint8)
        lens = [len(data[f"traj_{e['episode_id']}"]["actions"]) for e in self.episodes]
        first = data[f"traj_{self.episodes[0]['episode_id']}"]
        self.action_dim = first["actions"].shape[-1]
        self.obs = None
        if load_pixels:
            img_shape = first["obs"]["sensor_data"][camera]["rgb"].shape[1:]
            self.obs = np.empty((sum(lens) + len(lens), *img_shape), dtype=np.uint8)
        self.actions = np.empty((sum(lens), self.action_dim), dtype=np.float32)

        # per-episode (obs_start, act_start, n_windows). Episodes store eps_len + 1
        # observations but only eps_len actions, so the offsets are tracked separately.
        span = (num_steps - 1) * frameskip  # obs steps covered by a window
        obs_offset, act_offset = 0, 0
        windows = []
        for e, eps_len in zip(tqdm(self.episodes, desc="loading trajectories"), lens):
            traj = data[f"traj_{e['episode_id']}"]
            if load_pixels:
                self.obs[obs_offset : obs_offset + eps_len + 1] = traj["obs"]["sensor_data"][camera]["rgb"][: eps_len + 1]
            self.actions[act_offset : act_offset + eps_len] = traj["actions"][:]
            # a window starting at t needs obs t + span and actions up to t + span + frameskip - 1
            n_windows = eps_len - span - frameskip + 1
            if n_windows > 0:
                windows.append((obs_offset, act_offset, n_windows))
            obs_offset += eps_len + 1
            act_offset += eps_len
        data.close()

        self.episode_windows = np.array(windows, dtype=np.int64)
        self.action_mean = np.zeros(self.action_dim, dtype=np.float32)
        self.action_std = np.ones(self.action_dim, dtype=np.float32)

    def fit_mean_std(self, episode_indices):
        """z-score actions using statistics of the given episodes (e.g. the train split)."""
        rows = [
            np.arange(act_start, act_start + n + (self.num_steps * self.frameskip) - 1)
            for _, act_start, n in self.episode_windows[np.asarray(episode_indices)]
        ]
        sample = self.actions[np.concatenate(rows)]
        self.action_mean = sample.mean(0)
        self.action_std = np.maximum(sample.std(0), 1e-6)

    def all_action_windows(self, episode_indices=None):
        """every action window of the given episodes (default: all), normalized like __getitem__.
        Returns (N, num_steps, frameskip * action_dim) float32."""
        rows = self.episode_windows if episode_indices is None else self.episode_windows[np.asarray(episode_indices)]
        t = np.concatenate([act_start + np.arange(n) for _, act_start, n in rows])  # window start rows
        steps = np.arange(self.num_steps) * self.frameskip
        act_idx = t[:, None, None] + steps[:, None] + np.arange(self.frameskip)  # (N, T, fs)
        action = (self.actions[act_idx] - self.action_mean) / self.action_std
        return action.reshape(len(t), self.num_steps, -1).astype(np.float32)

    def __len__(self):
        # one item per episode (episodes too short for one window are skipped)
        return len(self.episode_windows)

    def __getitem__(self, idx):
        obs_start, act_start, n = self.episode_windows[idx]
        if n >= self.num_windows:
            t = np.random.permutation(n)[: self.num_windows]
        else:  # episode too short for num_windows distinct windows
            t = np.random.randint(n, size=self.num_windows)
        t = np.sort(t)

        steps = np.arange(self.num_steps) * self.frameskip  # (T,)
        obs_idx = obs_start + t[:, None] + steps  # (W, T)
        # frame k's action = the frameskip raw actions taken between frame k and k+1
        act_idx = act_start + t[:, None, None] + steps[:, None] + np.arange(self.frameskip)  # (W, T, fs)

        action = (self.actions[act_idx] - self.action_mean) / self.action_std  # (W, T, fs, A)
        action = action.reshape(*action.shape[:2], -1)  # (W, T, fs * A)
        item = dict(action=torch.from_numpy(action).float())
        if self.load_pixels:
            item["pixels"] = torch.from_numpy(self.obs[obs_idx])  # (W, T, H, W, 3) uint8
        return item # , obs_start, act_start, self.episodes[idx]["elapsed_steps"]
