import optax
from flax import linen as nn

from src.models.lossfuns import cross_entropy_loss
from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances
from jax import numpy as jnp


class AutoEncoderStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def batch_dims(self) -> tuple:
        return (4, 2), 4

    def loss_fun(self):
        return cross_entropy_loss

    def init_writer(self) -> ModelWriter:
        return writer_instances["autoencoder"]


    def init_optim(self, learning_rate: float):
        return optax.adam(learning_rate)