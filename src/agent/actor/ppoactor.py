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
        self._target_model: ModelWrapper = ModelWrapper(model, "actor")
        self._polyak: float = hyperparameters["ddpg"]["polyak"]
        self._trainer = PPOActorTrainer(self._model.model)
        self._new_states = None
        self.key = random.PRNGKey(0)

    def approximate_best_action(self, states:  jnp.ndarray) -> jnp.ndarray:
        actions = self._model.forward(states)
        self.key, subkey = random.split(self.key)
        actions = rlax.add_gaussian_noise(subkey, actions, 1)
        return softmax_to_onehot(actions)

    def calculate_actions(self, new_states: jax.Array) -> jax.Array:
        self._new_states = new_states
        actions = self._target_model.forward(new_states)
        return softmax_to_onehot(actions)

    def update(self, states: jax.Array, advantage:  jax.Array):
        grads = self._trainer.train_step(self._model.params, self._new_states, advantage)
        self._model.apply_grads(grads)

    @property
    def model(self):
        return self._model

