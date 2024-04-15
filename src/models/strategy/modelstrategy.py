from abc import abstractmethod
import flax.linen as nn

from src.resultwriter import ModelWriter


class ModelStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def init_writer(self) -> ModelWriter:
        pass

    @abstractmethod
    def init_params(self, model: nn.Module) -> tuple:
        pass

    @abstractmethod
    def init_optim(self, learning_rate: float):
        pass

    @abstractmethod
    def loss_fun(self):
        pass

    # Trainer for more complex training procedures
    def init_trainer(self, model: nn.Module):
        pass
