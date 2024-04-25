from abc import abstractmethod

import jax

from src.pod.replaybuffer import ReplayBuffer


class WorldModelInterface:
    @abstractmethod
    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        pass

    @abstractmethod
    def reset(self) -> (jax.Array, float, bool, bool, dict):
        pass

    @abstractmethod
    def update(self, data: ReplayBuffer):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass



