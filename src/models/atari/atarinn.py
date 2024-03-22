import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.base.mlpatari import MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    second_input: int
    output_dimensions: int

    def setup(self):
        bottleneck = 1000
        self.cnn = CNNAtari(bottleneck)
        self.mlp = MLPAtari(bottleneck + self.second_input, self.output_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
        x = jnp.append(cnn, action, axis=-1)
        x = self.mlp(x)
        return x
