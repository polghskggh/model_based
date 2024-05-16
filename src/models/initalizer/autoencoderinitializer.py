import optax
from flax import linen as nn

from src.models.lossfuns import cross_entropy_loss
from src.models.initalizer.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances
from jax import numpy as jnp


class AutoEncoderInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(1))

    def batch_dims(self) -> tuple:
        return (4, 1), (3, 2)

    def loss_fun(self):
        return cross_entropy_loss

    def init_writer(self) -> ModelWriter:
        writer_instances["autoencoder"] = ModelWriter("world_model", ["autoencoder_loss"])
        return writer_instances["autoencoder"]
