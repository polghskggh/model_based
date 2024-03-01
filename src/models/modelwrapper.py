import numpy as np
from flax import linen as nn
from jax import numpy as jnp, Array
from jax import random as random
import optax
from jax import value_and_grad, vmap

from .lossfuns import loss_funs
from ..resultwriter import ModelWriter


class ModelWrapper:
    def __init__(self, model: nn.Module, model_writer: ModelWriter, learning_rate: float = 0.0005, loss="mse"):
        self._model = model
        self._params = model.init(random.PRNGKey(1), jnp.ones(model.input_dimensions))
        self._loss_fun = loss_funs[loss]
        self._optimizer = optax.adam(learning_rate=learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        self.model_writer = model_writer

    # batch loss
    def batch_loss(self, model, params, inputs, teacher_outputs) -> Array:
        loss_fun = vmap(self._loss_fun, (None, None, 0, 0))
        loss = jnp.mean(loss_fun(model, params, jnp.array(inputs), jnp.array(teacher_outputs)))
        return loss

    # forward pass + backwards pass
    def train_step(self, x: np.ndarray[float], y: np.ndarray[float]):
        loss, grads = value_and_grad(self.batch_loss, 1)(self._model, self._params, x, y)
        self.model_writer.add_data(loss)
        self.apply_grads(grads)

    def forward(self, x: np.ndarray[float]) -> np.ndarray[float] | float:
        return self._model.apply(self._params, x)

    # apply gradians to the model
    def apply_grads(self, grads: np.ndarray[float]):
        opt_grads, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params)
        self._params = optax.apply_updates(self._params, opt_grads)

    def update_polyak(self, rho: float, other_model):
        self._params = optax.incremental_update(self._params, other_model._params, rho)
