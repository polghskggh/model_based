import optax
from flax import linen as nn
import jax.numpy as jnp

from src.models.strategy.modelstrategy import ModelStrategy
from src.resultwriter import ModelWriter


class DDPGActorStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_writer(self) -> ModelWriter:
        return ModelWriter("critic", "critic_loss")

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def init_optim(self, learning_rate: float):
        label_function = DDPGActorStrategy.map_function_to_dictionary(lambda key, _: "none" if key == "cnn" else "adam")
        return optax.multi_transform(
            {'adam': optax.sgd(learning_rate), 'none': optax.set_to_zero()}, label_function)

    @staticmethod
    def map_function_to_dictionary(fun):
        def map_function(nested_dict):
            return {label: (map_function(node) if isinstance(node, dict) else fun(label, node))
                    for label, node in nested_dict.items()}

        return map_function

