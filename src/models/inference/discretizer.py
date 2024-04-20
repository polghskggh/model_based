import flax.linen as nn
import jax.lax
from jax import Array
from jax import numpy as jnp
from numpy import random
from jax import lax


class Discretizer(nn.Module):
    train: bool

    @nn.compact
    def __call__(self, continuous: Array):
        if not self.train:
            return Discretizer.v2_calculation(continuous)

        continuous = self._noise(continuous)
        latent = Discretizer.v1_calculation(continuous) # feed gradient through v1

        if random.rand() > 0.5:
            latent += Discretizer.v2_calculation(continuous)
            latent -= lax.stop_gradient(Discretizer.v1_calculation(continuous))

        return latent

    @staticmethod
    def _noise(x: Array) -> Array:
        return x + random.normal(size=x.shape)

    @staticmethod
    def v1_calculation(x: Array) -> Array:
        return jnp.maximum(jnp.minimum(1.2 * nn.sigmoid(x) - 0.1, 1), 0)

    @staticmethod
    def v2_calculation(x: Array) -> Array:
        return jnp.where(x < 0, 1.0, 0.0)

