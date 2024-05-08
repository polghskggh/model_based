from collections import deque

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from jax import lax

from src.pod.hyperparameters import hyperparameters


class FrameStack:
    def __init__(self, env: gym.Env):
        env_stack, _, = env.reset()
        self.initial_state = lax.slice_in_dim(env_stack, 0, 3, axis=-1)
        self.size = hyperparameters["world"]["frame_stack"]
        self._lazy_frames = None
        self._frames = None
        self.reset()

    def reset(self):
        self._lazy_frames = deque([], maxlen=self.size)
        for _ in range(self.size):
            self._lazy_frames.append(self.initial_state)
        return self.frames

    def add_frame(self, next_frame):
        self._lazy_frames.append(next_frame)
        self._frames = None

    @property
    def frames(self):
        if self._frames is not None:
            return self._frames

        frames = jnp.array(self._lazy_frames, dtype=jnp.float32)
        new_shape = frames.shape[1:3] + (frames.shape[0] * frames.shape[3],)
        self._frames = frames.transpose(1, 2, 0, 3).reshape(new_shape)
        return self._frames
