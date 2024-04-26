import jax.numpy as jnp
from flax import linen as nn

from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances


class PPOActorStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (2,)

    def init_writer(self) -> ModelWriter:
        writer_instances["actor"] = ModelWriter("actor", ["actor_loss"])
        return writer_instances["actor"]
