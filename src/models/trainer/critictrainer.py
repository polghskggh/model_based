from ctypes import Array

from jax import value_and_grad

from src.models.lossfuns import mean_squared_error
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer

import jax.numpy as jnp

from rlax import truncated_generalized_advantage_estimation

from src.pod.hyperparameters import hyperparameters


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
        self._discount_factor = hyperparameters["ppo"]["discount_factor"]
        self._lambda = hyperparameters["ppo"]["lambda"]

    def train_step(self, rewards, values):
        advantage = truncated_generalized_advantage_estimation(rewards, self._discount_factor, self._lambda, values)
        return advantage
