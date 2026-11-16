"""
Rollout buffer for online LeWM training: a fixed number of episode slots that is also the
training Dataset.

It starts out holding the train-split expert demos. A random `pin_frac` of them are pinned and
never evicted. Each new rollout (`add`) replaces an unpinned expert episode while any are left,
then the oldest rollout (FIFO). Items are sampled exactly like ManiSkillTrajectoryDataset:
one item = `num_windows` windows from one slot's episode, so train.py's collate/forward apply.

Slots are plain lists of per-episode arrays (no padding to a max length). Expert slots are
views into the source dataset's arrays, so filling the buffer copies nothing.
DataLoader workers fork a copy of the buffer: build a fresh (non-persistent) loader after
every `add`, or use num_workers=0.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import ManiSkillTrajectoryDataset

EXPERT, ROLLOUT = 0, 1


class RolloutBuffer(Dataset):
    def __init__(self, num_steps: int, frameskip: int, num_windows: int, action_mean, action_std, seed: int = 0):
        self.num_steps, self.frameskip, self.num_windows = num_steps, frameskip, num_windows
        self.action_mean = np.asarray(action_mean, dtype=np.float32)  # frozen: the planner uses them too
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.obs: list[np.ndarray] = []      # (L+1, h, w, 3) uint8 per slot
        self.actions: list[np.ndarray] = []  # (L, A) float32 raw actions per slot
        self.source = np.empty(0, np.int8)   # EXPERT / ROLLOUT
        self.pinned = np.empty(0, bool)
        self.added_at = np.empty(0, np.int64)  # insertion counter, for FIFO eviction
        self.seed = np.empty(0, np.int64)      # env seed the episode was collected with
        self.n_added = 0

    @classmethod
    def from_dataset(cls, ds: ManiSkillTrajectoryDataset, indices, num_windows: int, pin_frac: float, seed: int = 0):
        """fill with episodes `indices` of ds (e.g. the train split); pin a random pin_frac of them."""
        buf = cls(ds.num_steps, ds.frameskip, num_windows, ds.action_mean, ds.action_std, seed)
        indices = np.asarray(indices)
        for i in indices:
            o0, a0, n = ds.episode_windows[i]
            L = n + (ds.num_steps - 1) * ds.frameskip + ds.frameskip - 1  # episode length (actions)
            buf.obs.append(ds.obs[o0 : o0 + L + 1])
            buf.actions.append(ds.actions[a0 : a0 + L])
        N = len(indices)
        buf.source = np.full(N, EXPERT, np.int8)
        buf.pinned = np.zeros(N, bool)
        buf.pinned[buf.rng.permutation(N)[: int(round(pin_frac * N))]] = True
        buf.added_at = np.zeros(N, np.int64)
        buf.seed = np.array([ds.episodes[i]["episode_seed"] for i in indices], np.int64)
        return buf

    def n_windows(self, L: int) -> int:
        """windows in an episode of L actions (same rule as ManiSkillTrajectoryDataset)."""
        return L - (self.num_steps - 1) * self.frameskip - self.frameskip + 1

    def add(self, obs: np.ndarray, actions: np.ndarray, seed: int) -> int:
        """insert one episode, obs (L+1, h, w, 3) uint8 and raw actions (L, A). Returns the slot
        it replaced, or -1 if the episode is too short for a single window (not added)."""
        assert len(obs) == len(actions) + 1, (obs.shape, actions.shape)
        if self.n_windows(len(actions)) < 1:
            return -1
        free = ~self.pinned
        experts = np.flatnonzero(free & (self.source == EXPERT))
        if len(experts):
            slot = int(self.rng.choice(experts))
        else:
            cand = np.flatnonzero(free)
            slot = int(cand[self.added_at[cand].argmin()])
        self.n_added += 1
        self.obs[slot] = np.asarray(obs, dtype=np.uint8)
        self.actions[slot] = np.asarray(actions, dtype=np.float32)
        self.source[slot], self.added_at[slot], self.seed[slot] = ROLLOUT, self.n_added, seed
        return slot

    def stats(self) -> dict:
        return dict(
            expert_frac=float((self.source == EXPERT).mean()),
            rollouts=int((self.source == ROLLOUT).sum()),
            rollouts_added=self.n_added,
        )

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs, actions = self.obs[idx], self.actions[idx]
        n = self.n_windows(len(actions))
        if n >= self.num_windows:
            t = np.random.permutation(n)[: self.num_windows]  # np.random: train.seed_worker reseeds it per worker
        else:
            t = np.random.randint(n, size=self.num_windows)
        t = np.sort(t)
        steps = np.arange(self.num_steps) * self.frameskip
        obs_idx = t[:, None] + steps  # (W, T)
        act_idx = t[:, None, None] + steps[:, None] + np.arange(self.frameskip)  # (W, T, fs)
        action = (actions[act_idx] - self.action_mean) / self.action_std
        action = action.reshape(*action.shape[:2], -1)  # (W, T, fs * A)
        return dict(action=torch.from_numpy(action).float(), pixels=torch.from_numpy(obs[obs_idx]))
