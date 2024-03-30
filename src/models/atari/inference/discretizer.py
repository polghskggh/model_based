import flax.linen as nn
from jax import Array
from jax import random, vmap
from jax import numpy as jnp

class Discretizer(nn.Module):
    def setup(self):
        self.discretizer = nn.Dense(256)
        self.key = random.PRNGKey(0)

    def _noise(self, x: Array) -> Array:
        return x + random.normal(self.key, x.shape)

    @nn.compact
    def __call__(self, continuous: Array):
        continuous = self._noise(continuous)
        v1 = vmap(vmap(self.threshold))(continuous)
        v2 = continuous > jnp.zeros(continuous)

        if random.rand(self.key) > 0.5:
            return v1
        else:
            return v2



    @staticmethod
    def threshold(x: int) -> int:
        return max(0, min(1, 1.2 * nn.sigmoid(x) - 0.1))

    def test_mod(self):
        pass

