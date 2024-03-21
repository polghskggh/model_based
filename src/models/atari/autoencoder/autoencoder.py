import flax.linen as nn
from jax import Array

from src.models.atari.autoencoder.decoder import Decoder
from src.models.atari.autoencoder.encoder import Encoder
from src.models.atari.autoencoder.injector import Injector


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.features = 256
        self.kernel = (4, 4)
        self.strides = (2, 2)
        self.layers = 6
        self.encoder = Encoder(self.features, self.kernel, self.strides, self.layers)
        self.decoder = Decoder(self.features, self.kernel, self.strides, self.layers)
        self.injector = Injector()

    @nn.compact
    def __call__(self, image: Array, action: Array):
        encoded = self.encoder(image)
        encoded = self.injector(encoded, action)
        return self.decoder(encoded)
