import jax.numpy as jnp

from src.models.lossfuns import cross_entropy_loss, cross_entropy_with_kl_loss
from src.models.initalizer.modelstrategy import ModelStrategy


class StochasticAETrainerInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32),
                jnp.ones(model.third_input, dtype=jnp.float32))

    def batch_dims(self):
        return (4, 2, 4), (4, )

    def loss_fun(self):
        return cross_entropy_with_kl_loss


class StochasticInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def batch_dims(self):
        return (4, 2), (4, )

    def loss_fun(self):
        return cross_entropy_loss


class InferenceInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32),
                jnp.ones(model.third_input, dtype=jnp.float32))

    def batch_dims(self) -> tuple:
        return (4, 2, 4), (2, )


class BitPredictorInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return ()

    def batch_dims(self) -> tuple:
        return None, None


class LatentAEInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model):
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32),
                jnp.ones(model.latent, dtype=jnp.float32))

    def batch_dims(self):
        return (4, 2, 2), (4, )

    def loss_fun(self):
        return cross_entropy_loss
