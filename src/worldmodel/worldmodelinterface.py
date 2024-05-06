from abc import abstractmethod

import jax

from src.pod.trajectorystorage import TrajectoryStorage


class WorldModelInterface:
    @abstractmethod
    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        pass

    @abstractmethod
    def reset(self) -> (jax.Array, float, bool, bool, dict):
        pass

    @abstractmethod
    def update(self, data: TrajectoryStorage):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass



