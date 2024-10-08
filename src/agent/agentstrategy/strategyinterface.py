from abc import abstractmethod
import jax.numpy as jnp


class StrategyInterface:
    @abstractmethod
    def timestep_callback(self, old_state: jnp.ndarray, reward: float, new_state: jnp.ndarray,
                          done: bool, store_trajectory: bool):
        pass

    @abstractmethod
    def select_action(self, states: jnp.ndarray, store_trajectories: bool = True) -> int:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
