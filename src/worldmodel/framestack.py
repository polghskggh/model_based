import jax.numpy as jnp
import numpy as np

from src.enviroment import Shape
from src.pod.storage import TransitionStorage
from src.singletons.hyperparameters import Args
import jax.random as jr

from src.singletons.rng import Key


class FrameStack:
    def __init__(self, data: jnp.ndarray):
        self._initial_states = data
        self._frames = None
        self.n_channels = Shape()[0][2] // Args().args.frame_stack
        self.reset()

    @staticmethod
    def sample_initial(initial_observations: jnp.ndarray, parallel_envs: int):
        idx = jr.choice(Key().key(1), initial_observations.shape[0], (parallel_envs,), False)
        return initial_observations[idx]

    def reset(self):
        self._frames = self.sample_initial(self._initial_states, Args().args.num_agents)
        np.save("f1", self._frames[:, :, :, 0])
        np.save("f2", self._frames[:, :, :, 1])
        np.save("f3", self._frames[:, :, :, 2])
        np.save("f4", self._frames[:, :, :, 3])
        return self._frames

    def add_frame(self, next_frame):
        self._frames = jnp.roll(self._frames, -self.n_channels, axis=-1)
        self._frames = self._frames.at[:, :, :, -self.n_channels:].set(next_frame)

    @property
    def frames(self):
        return self._frames
