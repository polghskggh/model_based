from ctypes import Array

from jax import value_and_grad

from src.models.lossfuns import mean_squared_error
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer

import jax.numpy as jnp


class CriticTrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def train_step(self, params, states: Array[float], actions: Array[float]):
        grad_fun = value_and_grad(mean_squared_error, 3)
        q_val, grads = grad_fun(self._model, params, states, actions)
        print(q_val)
        return grads

    @staticmethod
    def compound_grad_asc(model_fixed, params_fixed, model_deriv, params_deriv, state):
        actions = model_deriv.apply(params_deriv, state)
        q_values = model_fixed.apply(params_fixed, state, actions)
        return jnp.mean(q_values)


