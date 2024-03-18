import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.base.mlpatari import MLPAtari


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    output_dimensions: int

    def setup(self):
        self.encoder = CNNAtari(self.input_dimensions[0], self.input_dimensions[1], self.input_dimensions[2], 10)
        self.decoder = MLPAtari(10 + self.second_input, self.output_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
        x = jnp.append(cnn, action, axis=-1)
        x = self.mlp(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        return x
