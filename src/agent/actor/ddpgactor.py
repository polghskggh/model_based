from .__init__ import ActorInterface
from src.models import ModelWrapper
from flax import linen as nn
from jax import vmap

import numpy as np

from ...resultwriter import ModelWriter


class DDPGActor(ActorInterface):
    def __init__(self, model: nn.Module, polyak: float = 0.995):
        super().__init__()
        writer = ModelWriter("actor", "actor_q_value")
        self._model: ModelWrapper = ModelWrapper(model, "actor")
        self._target_model: ModelWrapper = ModelWrapper(model, "actor")
        self._polyak: float = polyak

    def approximate_best_action(self, state: np.ndarray[float]) -> np.ndarray[float]:
        return self._model.forward(state)

    def calculate_actions(self, new_states: np.ndarray[float]) -> np.ndarray[float]:
        return self._target_model.forward(new_states)

    def update_model(self, state: np.ndarray[float], selected_actions: np.ndarray[float],
                     action_grads: np.ndarray[float]):
        self._model.train_step(selected_actions + action_grads, state)
        self._target_model.update_polyak(self._polyak, self._model)

    @property
    def model(self):
        return self._model
