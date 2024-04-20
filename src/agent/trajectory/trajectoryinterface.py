from abc import abstractmethod
import jax.numpy as jnp


class TrajectoryInterface:
    @abstractmethod
    def update_input(self):
        pass

    @abstractmethod
    def add_transition(self, state: jnp.ndarray, action: jnp.ndarray, reward: float, next_state: jnp.ndarray):
        pass
