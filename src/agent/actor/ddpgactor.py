from src.agent.actor.actorinterface import ActorInterface
from src.models.modelwrapper import ModelWrapper
from flax import linen as nn
from jax import vmap
import jax.random as random

import numpy as np


class DDPGActor(ActorInterface):
    def __init__(self, model: nn.Module, polyak: float = 0.995):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "actor")
        self._target_model: ModelWrapper = ModelWrapper(model, "actor")
        self._polyak: float = polyak
        print(self._model)

    def approximate_best_action(self, state: np.ndarray[float]) -> np.ndarray[float]:
        actions = self._model.forward(state)
        return DDPGActor.softmax_to_onehot(actions)

    def calculate_actions(self, new_states: np.ndarray[float]) -> np.ndarray[float]:
        actions = self._target_model.forward(new_states)
        return DDPGActor.softmax_to_onehot(actions)

    def update(self, grads: np.ndarray[float]):
        self._model.apply_grads(grads)
        self._target_model.update_polyak(self._polyak, self._model)

    @staticmethod
    def softmax_to_onehot(logits: np.ndarray[float]) -> np.ndarray[float]:
        key = random.PRNGKey(0)
        if logits.ndim == 1:
            idx = random.choice(key, logits.shape[-1], p=logits)
        else:
            idx = vmap(random.choice, (None, None, None, None, 0))(key, logits.shape[-1], (), True, logits)
        return np.eye(logits.shape[-1])[idx]

    @property
    def model(self):
        return self._model
