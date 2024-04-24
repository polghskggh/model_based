from ctypes import Array
from functools import partial

import jax
import rlax
from jax import value_and_grad, vmap, lax, jit
import jax.numpy as jnp

from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances


class PPOActorTrainer(Trainer):
    def __init__(self, model: ModelWrapper):
        super().__init__()
        self._model = model
        self._clip_threshold = hyperparameters["ppo"]["clip_threshold"]
        self._rng = ModelWrapper.make_rng_keys()

    def train_step(self, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array):
        grad_fun = value_and_grad(PPOActorTrainer.batch_ppo_grad, 1)
        loss, grads = grad_fun(self._model, params, states, advantage, action_index, self._clip_threshold,
                               self._rng)

        writer_instances["actor"].add_data(loss)
        grads = jax.tree_util.tree_map(lambda x: -x, grads)
        return grads

    @staticmethod
    def ppo_grad(model, params: dict, states: jax.Array, advantage: float, action_index: int, epsilon: float, rng: dict):
        policy = jit(model.apply(params, states, rngs=rng))
        prob = jnp.take_along_axis(policy, jnp.expand_dims(action_index, 0), axis=1)
        old_prob = lax.stop_gradient(prob)

        clipped_loss = jnp.minimum(prob / old_prob * advantage,
                                   jnp.clip(prob / old_prob, 1 - epsilon, 1 + epsilon) * advantage)
        return clipped_loss

    @staticmethod
    def batch_ppo_grad(model, params: dict, states: jax.Array, advantage: float, action_index: jax.Array,
                       epsilon: float, rng: dict):
        in_axes = (None, None, 0, 0, 0, None, None)
        batch_loss_fun = vmap(PPOActorTrainer.ppo_grad, in_axes=in_axes)
        batch_loss = batch_loss_fun(model, params, states, advantage, action_index, epsilon, rng)
        return jnp.mean(batch_loss)
