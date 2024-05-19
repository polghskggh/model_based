from flax import linen as nn
from jax import numpy as jnp

from src.models.initalizer.modelstrategy import ModelStrategy
from src.models.lossfuns import cross_entropy_loss


class AutoEncoderInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(1))

    def batch_dims(self) -> tuple:
        return (4, 1), (3, 1)

    def loss_fun(self):
        return cross_entropy_loss

