from ctypes import Array
from typing import Self

import numpy as np
import optax
from flax import linen as nn
from jax import random as random
from jax import value_and_grad

from src.models.lossfuns import compound_grad_asc
from src.models.strategy.modelstrategyfactory import model_strategy_factory


class ModelWrapper:
    def __init__(self, model: nn.Module, strategy: str, learning_rate: float = 0.0001):
        self._strategy = model_strategy_factory(strategy)
        self._model = model
        self._params = model.init(random.PRNGKey(1), *self._strategy.init_params(model))
        self._loss_fun = self._strategy.loss_fun()
        self._optimizer = self._strategy.init_optim(learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        self.model_writer = self._strategy.init_writer()

    # forward pass + backwards pass
    def train_step(self, y: Array[float], *x: Array[float]):
        loss, grads = value_and_grad(self._loss_fun, 1)(self._model, self._params, y, *x)
        print(loss)
        self.model_writer.add_data(loss)
        self.model_writer.save_episode()
        return grads

    def forward(self, *x: Array[float]) -> Array[float] | float:
        return self._model.apply(self._params, *x)

    # apply gradians to the model
    def apply_grads(self, grads: Array[float]):
        opt_grads, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params)
        self._params = optax.apply_updates(self._params, opt_grads)

    # differentiate model with respect to the parameters of another model.
    def actor_grads(self, other_model: Self, states: np.ndarray[float]) -> dict:
        grad_fun = value_and_grad(compound_grad_asc, 3)
        q_val, grads = grad_fun(self._model, self._params, other_model._model, other_model._params, states)
        other_model.model_writer.add_data(q_val)
        other_model.model_writer.save_episode()
        return grads

    def update_polyak(self, rho: float, other_model: Self):
        self._params = optax.incremental_update(self._params, other_model._params, rho)

    def __str__(self):
        return self._model.tabulate(random.PRNGKey(0), *self._strategy.init_params(self._model))

    @property
    def params(self):
        return self._params
