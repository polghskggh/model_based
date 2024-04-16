import flax.linen as nn
from jax import Array
from jax import numpy as jnp
from numpy import random


class Discretizer(nn.Module):
    backprop: bool

    @nn.compact
    def __call__(self, continuous: Array):
        continuous = self._noise(continuous)

        if self.differentiable():
            return Discretizer.threshold(continuous)
        else:
            return jnp.where(continuous < 0, 1, 0)

    @staticmethod
    def _noise(x: Array) -> Array:
        return x + random.normal(size=x.shape)

    @staticmethod
    def threshold(x: Array) -> Array:
        return jnp.maximum(jnp.minimum(1.2 * nn.sigmoid(x) - 0.1, 1), 0)

    def differentiable(self) -> bool:
        if self.backprop:
            return True

        return random.rand() > 0.5

