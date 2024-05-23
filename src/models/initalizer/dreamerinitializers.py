import jax.numpy as jnp
import flax.linen as nn

from src.models.initalizer.modelstrategy import ModelStrategy


class TransitionInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return jnp.ones((model.state_size,)), jnp.ones(1), jnp.ones((model.belief_size, ))

    def batch_dims(self) -> tuple:
        return (2, 1, 2), None


class RepresentationInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones((model.state_size, )), jnp.ones(1), jnp.ones((model.belief_size, )),
                jnp.ones(model.observation_shape))

    def batch_dims(self) -> tuple:
        return (2, 1, 2, 4), None


class ObservationInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return jnp.ones((model.state_size, )), jnp.ones((model.belief_size, ))

    def batch_dims(self) -> tuple:
        return (2, 2), None


class RewardInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return jnp.ones((model.state_size, )), jnp.ones((model.belief_size, ))

    def batch_dims(self) -> tuple:
        return (2, 2), None

