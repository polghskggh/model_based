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
    dropout: float = 0.15

    def setup(self):
        self.features = 128
        self.pixel_embedding = nn.Dense(features=self.features // 2, name="embedding_top")
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6
        self.discretizer = Discretizer(self.backprop)

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array):
        embedding_input = ConvolutionalInference.merge((stack, actions, next_frame), (0, None, 0), 2)
        embedded_image = self.pixel_embedding(embedding_input)
        embedded_image = nn.relu(embedded_image)

        convolutions = []
        for _ in range(self.layers):
            embedded_image = nn.Dropout(rate=self.dropout, deterministic=False)(embedded_image)
            embedded_image = nn.LayerNorm()(embedded_image)
            embedded_image = nn.Conv(features=self.features, kernel_size=self.kernel, strides=self.strides)(embedded_image)
            embedded_image = nn.relu(embedded_image)
            convolutions.append(embedded_image)

        top = convolutions[0].reshape(convolutions[0].shape[0], -1)
        bottom = convolutions[1].reshape(convolutions[1].shape[0], -1)

        mean = nn.Dense(self.features)(top)
        std = nn.Dense(self.features)(bottom)

        continuous = self.sample_normal(mean, std)
        discrete = self.discretizer(continuous)
        return discrete

    def sample_normal(self, mean, std):
        rng = self.make_rng('normal')
        return jr.normal(rng, std.shape) * std + mean

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
