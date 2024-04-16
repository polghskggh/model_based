import optax
from flax import linen as nn
import jax.numpy as jnp

from src.models.lossfuns import mean_squared_error
from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances


class DDPGActorStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return 2, 2

    def init_writer(self) -> ModelWriter:
        return writer_instances["actor"]
