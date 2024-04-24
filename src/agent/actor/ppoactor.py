import jax
import rlax
from numpy import ndarray
from rlax import one_hot

from src.agent.actor.actorinterface import ActorInterface
from src.models.modelwrapper import ModelWrapper
from flax import linen as nn

import jax.numpy as jnp
import jax.random as random

import numpy as np

from src.models.trainer.actortrainer import DDPGActorTrainer, PPOActorTrainer
from src.pod.hyperparameters import hyperparameters
from src.utils.inttoonehot import softmax_to_onehot


class PPOActor(ActorInterface):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "actor")
        self._trainer = PPOActorTrainer(self._model.model)
        self._new_states = None
        self.key = random.PRNGKey(0)

    def calculate_actions(self, states: jax.Array) -> jax.Array:
        return self._model.forward(states)

    def calculate_grads(self, states: jax.Array, advantage: jax.Array, action: jax.Array) -> dict:
        grads = self._trainer.train_step(self._model.params, states, advantage, action)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    @property
    def model(self):
        return self._model

    def save(self):
        self._model.save("actor")

    def load(self):
        self._model.load("actor")


