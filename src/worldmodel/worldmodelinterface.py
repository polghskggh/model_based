from abc import abstractmethod

import jax



class WorldModelInterface:
    @abstractmethod
    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        pass

    @abstractmethod
    def reset(self) -> (jax.Array, dict):
        pass

    @abstractmethod
    def update(self, data):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def wrap_env(self, env):
        return env


