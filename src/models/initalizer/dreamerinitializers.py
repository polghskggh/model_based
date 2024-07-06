import jax.numpy as jnp
import flax.linen as nn

from src.enviroment import Shape
from src.models.initalizer.modelstrategy import ModelStrategy
from src.singletons.hyperparameters import Args


class TransitionInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return jnp.ones((Args().args.state_size,)), jnp.ones(1), jnp.ones((Args().args.belief_size, ))

    def batch_dims(self) -> tuple:
        return (2, 1, 2), None


class RepresentationInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones((Args().args.state_size, )), jnp.ones(1), jnp.ones((Args().args.belief_size, )),
                jnp.reshape(jnp.ones(Args().args.bottleneck_dims), -1))

    def batch_dims(self) -> tuple:
        return (2, 1, 2, 2), None


class ObservationInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return jnp.ones((Args().args.state_size, )), jnp.ones((Args().args.belief_size, ))

    def batch_dims(self) -> tuple:
        return (2, 2), None


class RewardInitializer(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return jnp.ones((Args().args.state_size, )), jnp.ones((Args().args.belief_size, ))

    def batch_dims(self) -> tuple:
        return (2, 2), None


class EncoderInitializer(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(Shape()[0]), )

    def batch_dims(self) -> tuple:
        return (4, ), None
