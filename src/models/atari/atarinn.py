import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari import CNNAtari, MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    action_size: int
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions[0], self.input_dimensions[1], self.input_dimensions[2], 10)
        self.mlp = MLPAtari(10 + self.action_size, self.output_dimensions)

    def __call__(self, x: Array):
        x[0] = self.cnn(x[0])
        x = self.mlp(x)
        return x



