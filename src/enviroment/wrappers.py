from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import WrapperObsType, WrapperActType
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper, Wrapper
import gymnasium as gym
from gymnasium.spaces.discrete import Discrete


class ReshapeObservation(ObservationWrapper):
    def __init__(self, env) -> gym.Env:
        super().__init__(env)
        self.new_shape = (env.observation_space.shape[1:3] +
                          (env.observation_space.shape[0] * env.observation_space.shape[3],))
        self.transpose = (1, 2, 0, 3)
        low = env.observation_space.low.transpose(*self.transpose).reshape(self.new_shape)
        high = env.observation_space.high.transpose(*self.transpose).reshape(self.new_shape)
        self.observation_space = Box(low=low, high=high, shape=self.new_shape,
                                     dtype=env.observation_space.dtype)

    def observation(self, observation: "LazyFrames"):
        """
        transform observation from (Stack, Height, Width, Channel)
                                to (Height, Width, Channel * Stack)

        :param observation: observation from the environment
        :return: reshaped observation
        """
        return np.array(observation).transpose(*self.transpose).reshape(self.new_shape)


class FrameSkip(Wrapper):
    def __init__(self, env: gym.Env, skip: int):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action: int):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, trunk, info


class CompatibilityWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = Box(low=env.observation_space.low, high=env.observation_space.high,
                                     shape=env.observation_space.shape, dtype=env.observation_space.dtype)
        self.action_space = Discrete(n=env.action_space.n)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        return self.env.reset()

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action.item())
