from typing import Optional
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from collections import deque

import torch

class RGBFrameStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'PickCube-v1', the original observation
    is an array with shape [42], so if we stack 4 observations, the processed observation
    has shape [4, 42].

    This wrapper also supports dict observations, and will stack the leafs of the dictionary accordingly.

    Note:
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.
    """

    def __init__(self, env: gym.Env, num_stack: int, frameskip: int = 1, lz4_compress: bool = False):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            frameskip (int): env steps between consecutive stacked frames (matches the
                world model's frameskip, so the stack is [o_{t-(n-1)k}, ..., o_{t-k}, o_t])
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.frameskip = frameskip
        # keep every raw frame in the window and subsample every `frameskip`-th one
        self.frames = deque(maxlen=(num_stack - 1) * frameskip + 1)
        self.lz4_compress = lz4_compress

        [self.frames.append(self.base_env._init_raw_obs["sensor_data"]["base_camera"]["rgb"].clone()) for _ in range(self.frames.maxlen)]
        self.use_dict = isinstance(self.base_env._init_raw_obs["sensor_data"]["base_camera"]["rgb"], dict)
        new_obs = self.observation(self.base_env._init_raw_obs["sensor_data"]["base_camera"]["rgb"])
        self.base_env.update_obs_space(new_obs)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation):
        assert len(self.frames) == self.frames.maxlen, (len(self.frames), self.frames.maxlen)
        frames = list(self.frames)[:: self.frameskip]
        if self.use_dict:
            return {
                k: torch.stack([x[k] for x in frames], dim=0).transpose(0, 1)
                for k in self.observation_space.keys()
            }
        else:
            return torch.stack(frames).transpose(0, 1)

        # LazyFrames equivalent code. It is unclear yet if this is faster or saves much memory. LazyFrames leverages __slots__ to save memory
        # and have faster attribute access.
        # if self.use_dict:
        #     return {
        #         k: LazyFrames([x[k] for x in self.frames], self.lz4_compress)
        #         for k in self.observation_space.keys()
        #     }
        # else:
        #     return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.frames.append(observation["sensor_data"]["base_camera"]["rgb"].clone())  # sensor buffers are reused in place
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment with kwargs.

        Args:
            seed: The seed for the environment reset
            options: The options for the environment reset

        Returns:
            The stacked observations
        """
        if (
            isinstance(options, dict)
            and "env_idx" in options
            and len(options["env_idx"]) < self.base_env.num_envs
        ):
            raise RuntimeError(
                "partial environment reset is currently not supported for the FrameStack wrapper at this moment for GPU parallelized environments"
            )
        # NOTE (stao): we can support partial reset easily using tensordicts
        obs, info = self.env.reset(seed=seed, options=options)
        [self.frames.append(obs["sensor_data"]["base_camera"]["rgb"].clone()) for _ in range(self.frames.maxlen)]

        return self.observation(None), info


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = RGBFrameStack(env, num_stack=other_kwargs["obs_horizon"], frameskip=other_kwargs.get("frameskip", 1))
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="LeWorldModel",
                        source_desc="LeWorldModel evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = RGBFrameStack(env, num_stack=other_kwargs["obs_horizon"], frameskip=other_kwargs.get("frameskip", 1))
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="leworldmodel",
                source_desc="leworldmodel evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
