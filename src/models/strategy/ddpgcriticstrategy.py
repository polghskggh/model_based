import optax
from flax import linen as nn
import jax.numpy as jnp

from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter


class DDPGCriticStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_writer(self):
        return ModelWriter("critic", "critic_loss")

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def init_optim(self, learning_rate: float):
        return optax.adam(learning_rate)
