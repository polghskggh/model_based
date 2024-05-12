from ctypes import Array

import jax
import optax
from flax import linen as nn
from jax import random as random, jit
from jax import value_and_grad
import orbax.checkpoint

from src.models.strategy.modelstrategyfactory import model_strategy_factory
from src.pod.hyperparameters import hyperparameters
from src.utils.transformtobatch import transform_to_batch


class ModelWrapper:
    def __init__(self, model: nn.Module, strategy: str, learning_rate: float = 0.0001):
        self._strategy = model_strategy_factory(strategy)
        self._model = model
        self._rngs = ModelWrapper.make_rng_keys()
        self._params = model.init(self._rngs, *self.batch_input(*self._strategy.init_params(model)))
        self._loss_fun = self._strategy.loss_fun()
        self._optimizer = self._strategy.init_optim(learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        self.model_writer = self._strategy.init_writer()
        self._version = 0

    # forward pass + backwards pass
    def train_step(self, y: jax.Array | tuple, *x: jax.Array) -> dict:
        """
        Train the model using a single step

        :param y: the target output
        :param x: the model input
        :return: the gradients of the model parameters
        """
        in_dim, out_dim = self._strategy.batch_dims()
        x = self.batch(x, in_dim)
        y = self.batch(y, out_dim)
        grad_fun = value_and_grad(self._loss_fun, 1)
        loss, grads = grad_fun(self._model, self._params, y, *x, rngs=self._rngs)
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
        return jit(self._model.apply)(self._params, *x, rngs=self._rngs)

    def apply_grads(self, grads: dict):
        """
        Apply gradients to the model using the optimizer

        :param grads: the gradients to apply
        """
        opt_grads, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params)
        self._params = optax.apply_updates(self._params, opt_grads)

    @property
    def model(self):
        """
        Get the model

        :return: the model
        """
        return self._model

    @property
    def params(self):
        """
        Get the parameters of the model

        :return: the parameters of the model
        """
        return self._params

    @params.setter
    def params(self, params):
        """
        Set the parameters of the model

        :param params: the parameters to set
        """
        self._params = params

    def batch_input(self, *data):
        """
        batches the input of the model

        :param data: the input to the model
        :return: batched input
        """
        in_dim, _ = self._strategy.batch_dims()
        return self.batch(data, in_dim)

    @staticmethod
    def batch(data, dims):
        """
        Batch the data

        :param data: the data to batch
        :param dims: the batch dimensions of the data
        :return: batched data
        """
        if dims is None:
            return data

        if not isinstance(data, tuple):
            return transform_to_batch(data, dims[0])

        return tuple(transform_to_batch(datum, dim) for datum, dim in zip(data, dims))

    def save(self, path: str):
        """
        Save the model to a checkpoint

        :param path: the path to save the model
        """
        checkpoint = {"params": self._params, "opt_state": self._opt_state, "rngs": self._rngs}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        path = hyperparameters["save_path"] + "/" + path + str(self._version)
        orbax_checkpointer.save(path, checkpoint)
        self._version += 1

    def load(self, path: str):
        """
        Load the model from a checkpoint

        :param path: the path to the checkpoint
        """
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        path = hyperparameters["save_path"] + path
        checkpoint = orbax_checkpointer.restore(path)
        self._params = checkpoint["params"]
        self._opt_state = checkpoint["opt_state"]
        self._rngs = checkpoint["rngs"]

    def __str__(self):
        """
        String representation of the model

        :return: the string representation of the model architecture
        """
        return self._model.tabulate(random.PRNGKey(0), *self.batch_input(*self._strategy.init_params(self._model)),
                                    console_kwargs={"width": 120})

    @staticmethod
    def make_rng_keys():
        return {"dropout": random.PRNGKey(hyperparameters["rng"]["dropout"]),
                "normal": random.PRNGKey(hyperparameters["rng"]["normal"]),
                "carry": random.PRNGKey(hyperparameters["rng"]["carry"]),
                "params": random.PRNGKey(hyperparameters["rng"]["params"])}
