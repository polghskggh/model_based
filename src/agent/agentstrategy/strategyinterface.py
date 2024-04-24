from abc import abstractmethod
import jax.numpy as jnp


class StrategyInterface:
    @abstractmethod
    def update(self, *data):
        pass

    @abstractmethod
    def action_policy(self, state: jnp.ndarray):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
