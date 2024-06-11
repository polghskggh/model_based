import os
from ctypes import Array

import jax
import optax
from flax import linen as nn
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax import random as random, jit
from jax import value_and_grad
import orbax.checkpoint
from orbax.checkpoint.type_handlers import ArrayHandler

from src.models.initalizer.modelstrategyfactory import model_initializer_factory
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer, log
from src.utils.modelhelperfuns import transform_to_batch
from src.utils.save_name import save_name


class ModelWrapper:
    def __init__(self, model: nn.Module, strategy: str, train_model: nn.Module = None):
        self._initializer = model_initializer_factory(strategy)
        self._model = model
        self._rngs = ModelWrapper.make_rng_keys()
        params = model.init(self._rngs, *self.batch_input(*self._initializer.init_params(model)))

        optimizer = self._initializer.init_optim()
        self.state = TrainState.create(apply_fn=jit(model.apply), params=params, tx=optimizer)

        self._loss_fun = self._initializer.loss_fun()
        # As we have separated agent and critic we don't use apply_fn

        self._name = strategy
        self._model_writer = Writer().writer
        self._train_model = train_model if (train_model is not None and Args().args.dropout) else self._model

    # forward pass + backwards pass
    def train_step(self, y: jax.Array | tuple, *x: jax.Array) -> dict:
        """
        Train the model using a single step

        :param y: the target output
        :param x: the model input
        :return: the gradients of the model parameters
        """
        in_dim, out_dim = self._initializer.batch_dims()
        x = self.batch(x, in_dim)
        y = self.batch(y, out_dim)

        grad_fun = value_and_grad(self._loss_fun, 1)

        loss, grads = grad_fun(self.state, self.state.params, y, *x, rngs=self._rngs)
        self._model_writer.add_scalar(f"losses/{self._name}_loss", loss, int(StepTracker()))

        return grads

    # forward pass
    def forward(self, *x: Array[float]) -> Array[float]:
        """
        Forward pass through the model

        :param x: the input to the model
        :return: the output of the model
        """
        x = self.batch_input(*x)
        return self.state.apply_fn(self.state.params, *x, rngs=self._rngs)

    def apply_grads(self, grads: dict):
        """
        Apply gradients to the model using the optimizer

        :param grads: the gradients to apply
        """
        self.state = self.state.apply_gradients(grads=grads)
        self._model_writer.add_scalar("charts/learning_rate",
                                      self.state.opt_state[1].hyperparams["learning_rate"].item(),
                                      int(StepTracker()))

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
        return self.state.params

    @params.setter
    def params(self, params):
        """
        Set the parameters of the model

        :param params: the parameters to set
        """
        self.state = self.state.replace(params=params)

    def batch_input(self, *data):
        """
        batches the input of the model

        :param data: the input to the model
        :return: batched input
        """
        in_dim, _ = self._initializer.batch_dims()
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

    def save(self):
        """
        Save the model to a checkpoint
        """
        ckpt = {'model': self.state, 'config': vars(Args().args)}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = ArrayHandler()
        checkpoint_dir = f'./model/{self._name}/{save_name()}'
        absolute_checkpoint_dir = os.path.abspath(checkpoint_dir)
        orbax_checkpointer.save(absolute_checkpoint_dir, ckpt, save_args=save_args)

    def load(self, path: str):
        """
        Load the model from a checkpoint

        :param path: the path to the checkpoint
        """
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint_dir = f'./model/{self._name}/{path}'
        absolute_checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.state, Args().args = orbax_checkpointer.restore(absolute_checkpoint_dir)

    def __str__(self):
        """
        String representation of the model

        :return: the string representation of the model architecture
        """
        return self._model.tabulate(random.PRNGKey(0), *self.batch_input(*self._initializer.init_params(self._model)),
                                    console_kwargs={"width": 120})

    @staticmethod
    def make_rng_keys():
        return {key: value for key, value in zip(["dropout", "normal", "carry", "params"], Key().key(4))}