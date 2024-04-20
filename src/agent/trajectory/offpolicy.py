from jax import numpy as jnp

from src.agent.trajectory.trajectoryinterface import TrajectoryInterface
from src.enviroment.shape import Shape
from src.pod import ReplayBuffer


class OffPolicy(TrajectoryInterface):
    def __init__(self):
        self._replay_buffer: ReplayBuffer = ReplayBuffer(*Shape())
        self._batch_size: int = 100

    def update_input(self):
        return self._replay_buffer.sample(self._batch_size)

    def add_transition(self, state: jnp.ndarray, action: jnp.ndarray, reward: float, next_state: jnp.ndarray):
        self._replay_buffer.add_transition(state, action, reward, next_state)


