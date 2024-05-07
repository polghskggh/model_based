import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax import Array, vmap

from src.models.inference.discretizer import Discretizer
from src.utils.activationfuns import activation_function_dict


class ConvolutionalInference(nn.Module):
    input_dimensions: tuple
    second_input: int
    third_input: tuple
    train: bool
    dropout: float = 0.15
    activation_function_name: str = 'relu'

    def setup(self):
        self.features = 128
        self.pixel_embedding = nn.Dense(features=self.features // 2, name="embedding_top")
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6
        self.discretizer = Discretizer(self.train)
        self.deterministic = not self.train
        self.activation_fun = activation_function_dict[self.activation_function_name]

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array, calculate_kl_loss: bool = False):
        embedding_input = ConvolutionalInference.merge((stack, actions, next_frame), (0, None, 0), 2)
        embedded_image = self.pixel_embedding(embedding_input)
        embedded_image = self.activation_fun(embedded_image)

        convolutions = []
        for _ in range(self.layers):
            embedded_image = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(embedded_image)
            embedded_image = nn.LayerNorm()(embedded_image)
            embedded_image = nn.Conv(self.features, kernel_size=self.kernel, strides=self.strides)(embedded_image)
            embedded_image = self.activation_fun(embedded_image)
            convolutions.append(embedded_image)

        top = convolutions[0].reshape(convolutions[0].shape[0], -1)
        bottom = convolutions[1].reshape(convolutions[1].shape[0], -1)

        mean = nn.Dense(self.features, name="mean_dist")(top)
        log_var = nn.Dense(self.features, name="log_std")(bottom)
        std = jnp.exp(log_var / 2.0)

        continuous = self.sample_normal(self.make_rng('normal'), mean, std)
        discrete = self.discretizer(continuous)
        nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(discrete)

        if not calculate_kl_loss:
            return discrete

        distribution = distrax.MultivariateNormalDiag(mean, std)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(jnp.zeros_like(mean), jnp.ones_like(std)))
        return discrete, kl_loss



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
