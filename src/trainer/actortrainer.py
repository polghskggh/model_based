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
        self._regularization = hyperparameters["ppo"]["regularization"]
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
        PPOActorTrainer.debug(prob, action_index, advantage)

        loss = rlax.clipped_surrogate_pg_loss(prob / lax.stop_gradient(prob), advantage, epsilon)
        return loss

    @staticmethod
    def debug(prob, action_index, advantage):
        mean_p = [0, 0, 0, 0]
        mean_adv = [0, 0, 0, 0]
        counts = [0, 0, 0, 0]
        for p, a, adv in zip(prob, action_index, advantage):
            counts[a] += 1
            mean_p[a] += p
            mean_adv[a] += adv

        for i in range(4):
            mean_p[i] /= max(counts[i], 1)
            mean_adv[i] /= max(counts[i], 1)
            jax.debug.print("probs: {prob}, action: {index}, advantage: {adv}", prob=mean_p[i], index=i, adv=mean_adv[i])