import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.atari.actoratari import CNNAtari
from src.models.atari.base.mlpatari import MLPAtari
from src.models.atari.base.revcnn import RevCNN


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.bottleneck = (10, 10, 10)
        self.encoder = Decoder(self.input_dimensions, self.bottleneck)
        self.decoder = Encoder(10 + self.second_input, self.input_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
        x = jnp.append(cnn, action, axis=-1)
        x = self.decoder(x)
        return x
