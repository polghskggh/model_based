from collections import deque

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from jax import lax

from src.pod.hyperparameters import hyperparameters
from src.pod.trajectorystorage import TrajectoryStorage


class FrameStack:
    def __init__(self, data: TrajectoryStorage):
        self.size = hyperparameters["frame_stack"]
        self._initial_states = data.sample_states(hyperparameters["simple"]["parallel_agents"])
        self._lazy_frames = None
        self._frames = None
        self.reset()

    def reset(self):
        self._lazy_frames = deque([], maxlen=self.size)
        self._frames = None
        for _ in range(self.size):
            self._lazy_frames.append(self._initial_states)

        self._frames = None
        return self.frames

    def add_frames(self, next_frames):
        self._lazy_frames.append(next_frames)
        self._frames = None

    @property
    def frames(self):
        if self._frames is not None:
            return self._frames

        frames = jnp.array(self._lazy_frames, dtype=jnp.float32)
        new_shape = (frames.shape[1],) + frames.shape[2:4] + (frames.shape[0] * frames.shape[4],)
        self._frames = frames.transpose(1, 2, 3, 0, 4).reshape(new_shape)

        return self._frames
