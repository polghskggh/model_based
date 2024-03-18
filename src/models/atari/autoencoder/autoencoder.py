import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.autoencoder.decoder import Decoder
from src.models.atari.autoencoder.encoder import Encoder
from src.models.atari.base.mlpatari import MLPAtari


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.bottleneck = (13, 10, 9)
        self.encoder = Encoder(self.input_dimensions, self.bottleneck)
        self.decoder = Decoder(self.bottleneck, self.second_input, self.input_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: int):
        encoded = self.encoder(image)
        return self.decoder(encoded, action)
