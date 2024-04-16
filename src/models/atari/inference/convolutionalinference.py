import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
from jax import Array, vmap

from src.models.atari.inference.discretizer import Discretizer


class ConvolutionalInference(nn.Module):
    input_dimensions: tuple
    second_input: int
    third_input: tuple
    backprop: bool

    def setup(self):
        self.features = 128
        self.pixel_embedding = nn.Dense(features=self.features // 2, name="embedding_top")
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6
        self.discretizer = Discretizer(self.backprop)
        self.key = jr.PRNGKey(1)

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array):
        embedding_input = ConvolutionalInference.merge((stack, actions, next_frame), (0, None, 0), 2)
        embedded_image = self.pixel_embedding(embedding_input)
        embedded_image = nn.relu(embedded_image)

        conv = nn.Conv(features=self.features, kernel_size=self.kernel, strides=self.strides)(embedded_image)
        conv = nn.relu(conv)

        conv2 = nn.Conv(features=self.features, kernel_size=self.kernel, strides=self.strides)(conv)
        conv2 = nn.relu(conv2)

        top = conv.reshape(conv.shape[0], -1)
        bottom = conv.reshape(conv2.shape[0], -1)

        mean = nn.Dense(self.features)(top)
        std = nn.Dense(self.features)(bottom)

        continuous = self.sample_normal(mean, std)
        discrete = self.discretizer(continuous)
        return discrete

    def sample_normal(self, mean, std):
        return jr.normal(self.key, std.shape) * std + mean

    @staticmethod
    def merge(inputs, dims, axis):
        mapped_fun = ConvolutionalInference.concatenate

        for index in range(axis):
            mapped_fun = vmap(mapped_fun, in_axes=dims)

        mapped_fun = vmap(mapped_fun, in_axes=(0, 0, 0))
        return mapped_fun(*inputs)

    @staticmethod
    def concatenate(*arrays):
        return jnp.concatenate(arrays)
