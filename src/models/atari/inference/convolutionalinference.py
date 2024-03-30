import flax.linen as nn
from jax import Array
import jax.numpy as jnp

from src.models.atari.autoencoder.pixelembedding import PixelEmbedding
from src.models.atari.inference.discretizer import Discretizer
from src.models.atari.inference.scaledlinear import ScaledLinear


class ConvolutionalInference(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.pixel_embedding = PixelEmbedding(64)
        self.features = 256
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6
        self.discretizer = Discretizer()

    @nn.compact
    def __call__(self, image: Array):
        embedded_image = self.pixel_embedding(image)
        embedded_image = nn.relu(embedded_image)

        conv = nn.Conv(embedded_image, kernel_size=self.kernel, strides=self.strides)(embedded_image)
        conv2 = nn.Conv(conv, kernel_size=self.kernel, strides=self.strides)(conv)

        top = conv.reshape(conv.shape[0], -1)
        bottom = conv.reshape(conv2.shape[0], -1)

        top = nn.Dense(4)(top)
        bottom = nn.Dense(4)(bottom)

        continuous = jnp.append(top, bottom, axis=-1)
        discretiscrete = self.discretizer(continuous)
        return conv2

