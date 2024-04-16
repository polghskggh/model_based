import optax

from src.models.lossfuns import mean_squared_error
from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter.modelwriter import writer_instances

import jax.random as jr
import jax.numpy as jnp


class StochasticAETrainerStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32),
                jnp.ones(model.third_input, dtype=jnp.float32))

    def batch_dims(self):
        return (4, 2, 4), 4


class StochasticAEStratgy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def batch_dims(self):
        return (4, 2), 4


class InferenceStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32),
                jnp.ones(model.third_input, dtype=jnp.float32))

    def batch_dims(self) -> tuple:
        return (4, 2, 4), 2


class BitPredictorStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return model.initialize_carry(jr.PRNGKey(0), (1, )), jnp.ones(1)

    def batch_dims(self) -> tuple:
        return None, None
