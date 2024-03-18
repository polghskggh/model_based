import flax.linen as nn
from jax.lax import stop_gradient
from jax import Array
import jax.numpy as jnp

from src.models.atari.base.cnnatari import CNNAtari


class ActorAtari(nn.Module):
    input_dimensions: tuple
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions, 10)

    @nn.compact
    def __call__(self, x: Array):
        x = self.cnn(x)
        x = stop_gradient(x)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.softmax(x)
        return x

    def mock_input(self) -> tuple:
        return (jnp.ones(self.input_dimensions, dtype=jnp.float32),)


