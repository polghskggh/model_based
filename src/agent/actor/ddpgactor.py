import rlax
from numpy import ndarray
from rlax import one_hot

from src.agent.actor.actorinterface import ActorInterface
from src.models.modelwrapper import ModelWrapper
from flax import linen as nn
from jax import vmap
import jax.random as random

import numpy as np

from src.models.trainer.actortrainer import DDPGActorTrainer
from src.pod.hyperparameters import hyperparameters


class DDPGActor(ActorInterface):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "actor")
        self._target_model: ModelWrapper = ModelWrapper(model, "actor")
        self._polyak: float = hyperparameters["ddpg"]["polyak"]
        self._trainer = DDPGActorTrainer(self._model.model)
        self._new_states = None
        self.key = random.PRNGKey(0)

    def approximate_best_action(self, state: np.ndarray[float]) -> ndarray[ndarray[float]]:
        actions = self._model.forward(state)
        self.key, subkey = random.split(self.key)
        actions = rlax.add_gaussian_noise(subkey, actions, 1)
        return DDPGActor.softmax_to_onehot(actions)

    def calculate_actions(self, new_states: np.ndarray[float]) -> ndarray[ndarray[float]]:
        self._new_states = new_states
        actions = self._target_model.forward(new_states)
        return DDPGActor.softmax_to_onehot(actions)

    def update(self, action_grads: np.ndarray[float]):
        grads = self._trainer.train_step(self._model.params, self._new_states, action_grads)
        self._model.apply_grads(grads)
        self._target_model.update_polyak(self._polyak, self._model)

    @property
    def model(self):
        return self._model

