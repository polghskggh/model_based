import flax.linen as nn
from jax import Array, vmap
import jax.numpy as jnp
import jax.random as jr

from src.models.atari.autoencoder.pixelembedding import PixelEmbedding
from src.models.atari.inference.discretizer import Discretizer


class ConvolutionalInference(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.pixel_embedding = PixelEmbedding(64, (0, 0, None))
        self.features = 256
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6
        self.discretizer = Discretizer()
        self.key = jr.PRNGKey(1)

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array):
        embedded_image = self.pixel_embedding(stack, actions, next_frame)
        embedded_image = nn.relu(embedded_image)

        conv = nn.Conv(embedded_image, kernel_size=self.kernel, strides=self.strides)(embedded_image)
        conv2 = nn.Conv(conv, kernel_size=self.kernel, strides=self.strides)(conv)

        top = conv.reshape(conv.shape[0], -1)
        bottom = conv.reshape(conv2.shape[0], -1)

        mean = nn.Dense(128)(top)
        std = nn.Dense(128)(bottom)

        continuous = self.sample_normal(mean, std)
        discrete = self.discretizer(continuous)
        return discrete

    def sample_normal(self, mean, std):
        return jr.normal(self.key, std.shape) * std + mean

