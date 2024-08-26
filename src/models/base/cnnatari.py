import jax
from flax import linen as nn
from jax import Array
import jax.numpy as jnp

from src.models.autoencoder.encoder import Encoder


# A simple feed forward neural network
class CNNAtari(nn.Module):
    output_dimensions: int
    last: jax.Array = None
    deterministic: bool = True

    def setup(self):
        features = 256
        kernel = (4, 4)
        strides = (2, 2)
        layers = 6
        self.encoder = Encoder(features, kernel, strides, layers, self.deterministic)

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x, _ = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.relu(x)
        return x
