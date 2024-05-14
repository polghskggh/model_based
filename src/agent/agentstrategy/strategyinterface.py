from abc import abstractmethod
import jax.numpy as jnp


class StrategyInterface:
    @abstractmethod
    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray,
               term: bool, trunc: bool):
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

    @abstractmethod
    def run_parallel(self, parallel_agents):
        pass
