from flax import linen as nn
from jax import Array

from src.models.atari.autoencoder.encoder import Encoder


# A simple feed forward neural network
class CNNAtari(nn.Module):
    output_dimensions: int

    def setup(self):
        features = 256
        kernel = (4, 4)
        strides = (2, 2)
        layers = 6
        self.encoder = Encoder(features, kernel, strides, layers)

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x, skip = self.encoder(x)
        x = CNNAtari.flatten(x)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.relu(x)
        return x

    @staticmethod
    def flatten(x: Array) -> Array:
        return x.reshape(-1) if x.ndim == 3 else x.reshape((x.shape[0], -1))
