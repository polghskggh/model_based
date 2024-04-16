from ctypes import Array

from jax import value_and_grad

from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer

import jax.numpy as jnp


class CriticTrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def train_step(self, params, other_model: ModelWrapper, states: Array[float]):
        grad_fun = value_and_grad(CriticTrainer.compound_grad_asc, 3)
        q_val, grads = grad_fun(params, other_model.model, other_model.params, states)
        other_model.model_writer.add_data(q_val)
        other_model.model_writer.save_episode()
        return grads

    @staticmethod
    def compound_grad_asc(model_fixed, params_fixed, model_deriv, params_deriv, state):
        actions = model_deriv.apply(params_deriv, state)
        q_values = model_fixed.apply(params_fixed, state, actions)
        return jnp.mean(q_values)


