import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.mlpatari import MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    second_input: int
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions[0], self.input_dimensions[1], self.input_dimensions[2], 10)
        self.mlp = MLPAtari(10 + self.second_input, self.output_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
        x = self.mlp(action)
        x = jnp.append(cnn, x, axis=-1)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        return x
