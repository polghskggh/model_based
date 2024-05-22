import jax.numpy as jnp

from src.pod.storage import TransitionStorage
from src.singletons.hyperparameters import Args
import jax.random as jr

from src.singletons.rng import Key


class FrameStack:
    def __init__(self, data: TransitionStorage):
        self._initial_states = FrameStack.sample_initial(data, Args().args.num_agents)
        self._frames = jnp.zeros(self._initial_states.shape, dtype=jnp.float32)
        self.reset()

    @staticmethod
    def sample_initial(data: TransitionStorage, parallel_envs: int):
        idx = jr.choice(Key().key(1), data.observations.shape[0], (parallel_envs,), False)
        return data.observations[idx]

    def reset(self):
        self._frames = self._initial_states
        return self._frames

    def add_frame(self, next_frame):
        self._frames = jnp.roll(self._frames, -3, axis=-1)
        self._frames.at[:, :, :, -3:].set(next_frame)

    @property
    def frames(self):
        return self._frames
