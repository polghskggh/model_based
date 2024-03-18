import optax
from flax import linen as nn
import jax.numpy as jnp

from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances


class DDPGActorStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_writer(self) -> ModelWriter:
        return writer_instances["actor"]

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def init_optim(self, learning_rate: float):
        return optax.adam(learning_rate)
