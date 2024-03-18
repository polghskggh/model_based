import numpy as np
from flax import linen as nn
from jax import random as random
import optax
from jax import value_and_grad, grad

from src.models.lossfuns import loss_funs
from src.models.strategy.modelstrategyfactory import model_strategy_factory


class ModelWrapper:
    def __init__(self, model: nn.Module, strategy: str, learning_rate: float = 0.0001, loss="mse"):
        self._strategy = model_strategy_factory(strategy)
        self._model = model
        self._params = model.init(random.PRNGKey(1), *self._strategy.init_params(model))
        self._loss_fun = loss_funs[loss]
        self._optimizer = self._strategy.init_optim(learning_rate)
        self._opt_state = self._optimizer.init(self._params)
        self.model_writer = self._strategy.init_writer()
        self.debug = strategy

    # forward pass + backwards pass
    def train_step(self, y: np.ndarray[float], *x: np.ndarray[float]):
        loss, grads = value_and_grad(self._loss_fun, 1)(self._model, self._params, y, *x)
        self.model_writer.add_data(loss)
        self.model_writer.save_episode()
        self.model_writer.flush_all()
        if self.debug == "actor":
            print(grads["params"]["Dense_0"])
        self.apply_grads(grads)

    def forward(self, *x: np.ndarray[float]) -> np.ndarray[float] | float:
        return self._model.apply(self._params, *x)

    # apply gradians to the model
    def apply_grads(self, grads: np.ndarray[float]):
        opt_grads, self._opt_state = self._optimizer.update(grads, self._opt_state, self._params)
        self._params = optax.apply_updates(self._params, opt_grads)

    def calculate_gradian_ascent(self, input_idx: int, *x: np.ndarray[float]) -> np.ndarray[float]:
        return grad(loss_funs["grad_asc"], 2 + input_idx)(self._model, self._params, *x)

    def update_polyak(self, rho: float, other_model: "ModelWrapper"):
        self._params = optax.incremental_update(self._params, other_model._params, 1 - rho)

    def __str__(self):
        return self._model.tabulate(random.PRNGKey(0), *self._strategy.init_params(self._model))

    @property
    def params(self):
        return self._params
