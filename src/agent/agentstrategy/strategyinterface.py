from abc import abstractmethod
import jax.numpy as jnp


class StrategyInterface:
    @abstractmethod
    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        pass

    @abstractmethod
    def select_action(self, state: jnp.ndarray):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
