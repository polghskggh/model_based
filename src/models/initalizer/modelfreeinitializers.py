import jax.numpy as jnp
import optax
from flax import linen as nn

from src.models.initalizer.helper import linear_schedule
from src.models.initalizer.modelstrategy import ModelStrategy
from src.singletons.hyperparameters import Args


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
                jnp.ones(model.second_input, dtype=jnp.float32))

    def batch_dims(self) -> tuple:
        return (4, 2), (2, )


class CriticInitializer(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (2, )
