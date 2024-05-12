import flax.linen as nn

import jax
import jax.numpy as jnp
import rlax
from jax import value_and_grad, vmap, lax, jit

from src.models.modelwrapper import ModelWrapper
from src.trainer.trainer import Trainer
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances


class PPOActorTrainer(Trainer):
    def __init__(self, model: nn.Module):
        self._model = model
        self._clip_threshold = hyperparameters["ppo"]["clip_threshold"]
        self._rng = ModelWrapper.make_rng_keys()

    def train_step(self, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array):
        grad_fun = value_and_grad(PPOActorTrainer.ppo_grad, 1)
        loss, grads = grad_fun(self._model, params, states, advantage, action_index, self._clip_threshold,
                               self._rng)

        writer_instances["actor"].add_data(loss)
        return grads

    @staticmethod
    def ppo_grad(model, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array,
                 epsilon: float, rng: dict):
        policy = jit(model.apply)(params, states, rngs=rng)
        prob = jnp.take_along_axis(policy, jnp.expand_dims(action_index, 1), axis=1)
        prob = jnp.squeeze(prob)
        old_prob = lax.stop_gradient(prob)
        loss = rlax.clipped_surrogate_pg_loss(prob / old_prob, advantage, epsilon)
        return loss
