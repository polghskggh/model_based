from abc import abstractmethod
from typing import Tuple

import flax.linen as nn
import optax

from src.models.lossfuns import mean_squared_error
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances


class ModelStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def init_params(self, model: nn.Module) -> tuple:
        pass

    @abstractmethod
    def batch_dims(self) -> Tuple:
        pass

    def init_writer(self) -> ModelWriter:
        return writer_instances["mock"]

    def init_optim(self, learning_rate: float):
        return optax.adam(learning_rate)

    def loss_fun(self):
        return mean_squared_error

