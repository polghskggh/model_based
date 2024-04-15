import optax
from flax import linen as nn
import jax.numpy as jnp

from src.models.lossfuns import mean_squared_error
from src.models.strategy.modelstrategy import ModelStrategy
from src.models.trainer.critictrainer import CriticTrainer
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances


class DDPGCriticStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def loss_fun(self):
        return mean_squared_error

    def init_writer(self):
        return writer_instances["critic"]

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def init_optim(self, learning_rate: float):
        return optax.adam(learning_rate)

    def init_trainer(self, model: nn.Module):
        return CriticTrainer(model)
