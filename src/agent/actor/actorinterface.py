from abc import abstractmethod

import jax
import numpy as np

from src.models.modelwrapper import ModelWrapper


class ActorInterface:
    @abstractmethod
    def calculate_actions(self, states: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def calculate_grads(self, *args):
        pass

    @abstractmethod
    def update(self, grads: dict):
        pass

    @abstractmethod
    def model(self) -> ModelWrapper:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
