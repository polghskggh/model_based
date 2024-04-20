from abc import abstractmethod
import jax.numpy as jnp


class StrategyInterface:
    @abstractmethod
    def update(self, *data):
        pass

    @abstractmethod
    def select_action(self, state: jnp.ndarray):
        pass
