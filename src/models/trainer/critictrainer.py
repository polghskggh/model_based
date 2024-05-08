from ctypes import Array

import jax
from jax import value_and_grad, vmap

from src.models.lossfuns import mean_squared_error
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer

import jax.numpy as jnp

from rlax import truncated_generalized_advantage_estimation

from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances
from src.utils.rebatch import rebatch


class DDPGCriticTrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def train_step(self, params, states: Array[float], actions: Array[float]):
        grad_fun = value_and_grad(mean_squared_error, 3)
        q_val, grads = grad_fun(self._model, params, states, actions)
        return grads


class PPOCriticTrainer:
    def __init__(self, model):
        super().__init__()
        self._model = model

    def train_step(self, params, rewards, states):
        batch_size = min(hyperparameters["ppo"]["batch_size"], states.shape[0] + states.shape[1])
        rewards, states = rebatch(batch_size, rewards, states)

        grad_fn = jax.value_and_grad(PPOCriticTrainer.loss_fun, 1)
        loss, grad = grad_fn(self._model, params, rewards, states)

        writer_instances["critic"].add_data(loss)
        return grad

    @staticmethod
    def loss_fun(model, params, rewards, states):
        batch_loss = 0
        for reward_b, state_b in rewards, states:
            batch_loss += mean_squared_error(model, params, reward_b, state_b)

        batch_loss /= len(rewards)
        return batch_loss
