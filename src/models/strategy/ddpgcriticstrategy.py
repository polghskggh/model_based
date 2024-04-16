import jax.numpy as jnp
from flax import linen as nn

from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter.modelwriter import writer_instances


class DDPGCriticStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def batch_dims(self) -> tuple:
        return (4, 2), 2

    def init_writer(self):
        return writer_instances["critic"]
