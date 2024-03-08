import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.mlpatari import MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions[0][0], self.input_dimensions[0][1], self.input_dimensions[0][2], 10)
        self.mlp = MLPAtari(10 + self.input_dimensions[1], self.output_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
        x = self.mlp(action)
        x = jnp.append(cnn, x, axis=-1)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        return x

    def mock_input(self) -> tuple:
        return (jnp.ones(self.input_dimensions[0], dtype=jnp.float32),
                jnp.ones(self.input_dimensions[1], dtype=jnp.float32))
