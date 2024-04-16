from ctypes import Array
from typing import Self

import optax
from flax import linen as nn
from jax import random as random, vmap
from jax import value_and_grad

from src.models.strategy.modelstrategyfactory import model_strategy_factory
from src.utils.transformtobatch import transform_to_batch


class ModelWrapper:
    def __init__(self, model: nn.Module, strategy: str, learning_rate: float = 0.0001):
        self._strategy = model_strategy_factory(strategy)
        self._model = model
        self._params = model.init(random.PRNGKey(1), *self.batch_input(*self._strategy.init_params(model)))
        self._loss_fun = self._strategy.loss_fun()
        self._optimizer = self._strategy.init_optim(learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        self.model_writer = self._strategy.init_writer()

    # forward pass + backwards pass
    def train_step(self, y: Array[float], *x: Array[float]):
        in_dims, out_dims = self._strategy.batch_dims()
        x = self.batch_input(*x)
        y = transform_to_batch(y, out_dims)
        loss, grads = value_and_grad(self._loss_fun, 1)(self._model, self._params, y, *x)
        self.model_writer.add_data(loss)
        self.model_writer.save_episode()
        return grads

    # forward pass
    def forward(self, *x: Array[float]) -> Array[float]:
        x = self.batch_input(*x)
        return self._model.apply_strategy(self._params, *x)

    def batch_input(self, *x: Array[float]):
        in_dims, _ = self._strategy.batch_dims()
        if in_dims is None:
            return x

        return (transform_to_batch(input, in_dim) for input, in_dim in zip(x, in_dims))

    # apply gradians to the model
    def apply_grads(self, grads: Array[float]):
        opt_grads, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params)
        self._params = optax.apply_updates(self._params, opt_grads)

    def update_polyak(self, rho: float, other_model: Self):
        self._params = optax.incremental_update(self._params, other_model._params, rho)

    def __str__(self):
        return self._model.tabulate(random.PRNGKey(0), *self._strategy.init_params(self._model))

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
