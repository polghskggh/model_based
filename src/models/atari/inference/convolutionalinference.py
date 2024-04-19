import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import rlax
from distrax import MultivariateNormalDiag
from jax import Array, vmap

from src.models.atari.inference.discretizer import Discretizer


class ConvolutionalInference(nn.Module):
    input_dimensions: tuple
    second_input: int
    third_input: tuple
    train: bool
    dropout: float = 0.15

    def setup(self):
        self.features = 128
        self.pixel_embedding = nn.Dense(features=self.features // 2, name="embedding_top")
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6
        self.discretizer = Discretizer(self.train)
        self.deterministic = not self.train

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array, calculate_kl_loss: bool = False):
        embedding_input = ConvolutionalInference.merge((stack, actions, next_frame), (0, None, 0), 2)
        embedded_image = self.pixel_embedding(embedding_input)
        embedded_image = nn.relu(embedded_image)

        convolutions = []
        for _ in range(self.layers):
            embedded_image = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(embedded_image)
            embedded_image = nn.LayerNorm()(embedded_image)
            embedded_image = nn.Conv(features=self.features, kernel_size=self.kernel, strides=self.strides)(embedded_image)
            embedded_image = nn.relu(embedded_image)
            convolutions.append(embedded_image)

        top = convolutions[0].reshape(convolutions[0].shape[0], -1)
        bottom = convolutions[1].reshape(convolutions[1].shape[0], -1)

        mean = nn.Dense(self.features, name="mean_dist")(top)
        log_var = nn.Dense(self.features, name="log_std")(bottom)
        std = jnp.exp(log_var / 2.0)

        continuous = self.sample_normal(mean, std)
        discrete = self.discretizer(continuous)
        nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(discrete)

        distribution = distrax.MultivariateNormalDiag(mean, std)

        if not calculate_kl_loss:
            return discrete

        kl_loss = distribution.kl_divergence(MultivariateNormalDiag(jnp.zeros_like(mean), jnp.ones_like(std)))
        return discrete, kl_loss

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
