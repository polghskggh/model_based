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


class PPOCriticTrainer:
    def __init__(self, model):
        super().__init__()
        self._model = model

    def train_step(self, params, rewards, states):
        grad_fn = jax.value_and_grad(mean_squared_error, 1)
        loss, grad = grad_fn(self._model, params, rewards, states)

        writer_instances["critic"].add_data(loss)
        return grad