from ctypes import Array
from typing import Self

import optax
from flax.training import orbax_utils
from flax import linen as nn
from jax import random as random
from jax import value_and_grad
import orbax.checkpoint

from src.models.strategy.modelstrategyfactory import model_strategy_factory
from src.pod.hyperparameters import hyperparameters
from src.utils.transformtobatch import transform_to_batch


class ModelWrapper:
    def __init__(self, model: nn.Module, strategy: str, learning_rate: float = 0.0001):
        self._strategy = model_strategy_factory(strategy)
        self._model = model
        self._rngs = {"dropout": random.PRNGKey(hyperparameters["rng"]["dropout"]),
                      "normal": random.PRNGKey(hyperparameters["rng"]["normal"]),
                      "carry": random.PRNGKey(hyperparameters["rng"]["carry"]),
                      "params": random.PRNGKey(hyperparameters["rng"]["params"])}

        self._params = model.init(self._rngs, *self.batch_input(*self._strategy.init_params(model)))
        self._loss_fun = self._strategy.loss_fun()
        self._optimizer = self._strategy.init_optim(learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        self.model_writer = self._strategy.init_writer()
        self.debug = strategy

    # forward pass + backwards pass
    def train_step(self, y: Array[float], *x: Array[float]):
        in_dim, out_dim = self._strategy.batch_dims()
        x = self.batch(x, in_dim)
        y = self.batch(y, out_dim)

        loss, grads = value_and_grad(self._loss_fun, 1)(self._model, self._params, y, *x, rngs=self._rngs)
        self.model_writer.add_data(loss)
        return grads

    # forward pass
    def forward(self, *x: Array[float]) -> Array[float]:
        """
        Forward pass through the model

        :param x: the input to the model
        :return: the output of the model
        """
        x = self.batch_input(*x)
        return self._model.apply(self._params, *x, rngs=self._rngs)

    def apply_grads(self, grads: dict):
        """
        Apply gradients to the model using the optimizer

        :param grads: the gradients to apply
        """
        opt_grads, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params)
        self._params = optax.apply_updates(self._params, opt_grads)

    def update_polyak(self, rho: float, other_model: Self):
        """
        Update the parameters of the model using Polyak averaging

        :param rho: the averaging factor
        :param other_model: the model to average with
        :return: None
        """
        self._params = optax.incremental_update(self._params, other_model._params, rho)

    def __str__(self):
        return self._model.tabulate(random.PRNGKey(0), *self.batch_input(*self._strategy.init_params(self._model)))

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    def batch_input(self, *data):
        in_dim, _ = self._strategy.batch_dims()
        return self.batch(data, in_dim)

    @staticmethod
    def batch(data, dims):
        if dims is None:
            return data

        if len(dims) == 1:
            return transform_to_batch(data, dims[0])

        return tuple(transform_to_batch(datum, dim) for datum, dim in zip(data, dims))

    def save(self, path: str):
        checkpoint = {"params": self._params, "opt_state": self._opt_state, "rngs": self._rngs}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        path = hyperparameters["save_path"] + path

        orbax_checkpointer.save(path, checkpoint, save_args=save_args)
        self._model.save(self._params)

    def load(self, path: str):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        path = hyperparameters["save_path"] + path
        checkpoint = orbax_checkpointer.restore(path)
        self._params = checkpoint["params"]
        self._opt_state = checkpoint["opt_state"]
        self._rngs = checkpoint["rngs"]
