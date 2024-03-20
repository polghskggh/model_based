import flax.linen as nn
from jax import Array

from src.models.atari.autoencoder.decoder import Decoder
from src.models.atari.autoencoder.encoder import Encoder


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
