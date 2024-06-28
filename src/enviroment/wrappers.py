from typing import Any, SupportsFloat

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.core import WrapperObsType, WrapperActType, ActionWrapper, RewardWrapper
from gymnasium.spaces import Box
from gymnasium.spaces.discrete import Discrete


class LimitActions(ActionWrapper):
    def __init__(self, env) -> gym.Env:
        super().__init__(env)
        self.action_space = Discrete(n=2)

    def action(self, action: jnp.ndarray) -> jnp.ndarray:
        return action + 1


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
        serve = jnp.array(observation).transpose(*self.transpose).reshape(self.new_shape)
        return serve


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
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        total_reward = np.clip(total_reward, -6.0, 6.0)
        return observation, total_reward, terminated, truncated, info


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


class CategoricalReward(RewardWrapper):
    def reward(self, reward: float) -> float:
        if reward > 0:
            return 1.0
        else:
            return 0.0

