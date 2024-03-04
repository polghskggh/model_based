import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.mlpatari import MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    action_size: int
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions[0] - self.action_size,
                            self.input_dimensions[1], self.input_dimensions[2], 10)
        self.mlp = MLPAtari(10 + self.action_size, self.output_dimensions)

    def __call__(self, x: Array):
        cnn = self.cnn(x[:, :self.cnn.xdim])
        action = x[:, self.cnn.xdim:]
        action = action.reshape((-1))[:self.action_size]
        x = self.mlp(jnp.append(cnn, action))
        return x



