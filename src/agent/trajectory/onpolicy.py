from jax import numpy as jnp

from src.agent.trajectory.trajectoryinterface import TrajectoryInterface


class OnPolicy(TrajectoryInterface):
    def __init__(self):
        pass

    def update_input(self):
        pass

    def add_transition(self, state: jnp.ndarray, action: jnp.ndarray, reward: float, next_state: jnp.ndarray):
        pass

