import jax.numpy as jnp
from flax import linen as nn

from src.models.initalizer.modelstrategy import ModelStrategy


class ActorCriticInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (1, )


class DQNInitializer(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(1))

    def batch_dims(self) -> tuple:
        return (4, 1), (2, )


class CriticInitializer(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (2, )
