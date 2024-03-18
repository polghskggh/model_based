from .__init__ import ActorInterface
from src.models.modelwrapper import ModelWrapper
from flax import linen as nn
from jax import vmap

import numpy as np

from ...resultwriter import ModelWriter


class DDPGActor(ActorInterface):
    def __init__(self, model: nn.Module, polyak: float = 0.995):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "actor")
        self._target_model: ModelWrapper = ModelWrapper(model, "actor")
        self._polyak: float = polyak

    def approximate_best_action(self, state: np.ndarray[float]) -> np.ndarray[float]:
        actions = self._model.forward(state)
        print(DDPGActor.softmax_to_onehot(actions))
        return actions

    def calculate_actions(self, new_states: np.ndarray[float]) -> np.ndarray[float]:
        actions = self._target_model.forward(new_states)
        print(DDPGActor.softmax_to_onehot(actions))
        return actions


    def update_model(self, state: np.ndarray[float], selected_actions: np.ndarray[float],
                     action_grads: np.ndarray[float]):
        self._model.train_step(selected_actions + action_grads, state)
        self._target_model.update_polyak(self._polyak, self._model)

    @staticmethod
    def softmax_to_onehot(logits: np.ndarray[float]) -> np.ndarray[float]:
        return np.eye(logits.shape[-1])[logits.argmax(axis=-1)]

    @property
    def model(self):
        return self._model
