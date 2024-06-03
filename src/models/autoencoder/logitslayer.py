from flax import linen as nn
from jax import Array, vmap
import jax.numpy as jnp


# logits
class LogitsLayer(nn.Module):
    def setup(self):
        self.features = 256
        self.red = nn.Dense(features=self.features, name="red_logits")
        self.green = nn.Dense(features=self.features, name="greed_logits")
        self.blue = nn.Dense(features=self.features, name="blue_logits")

    @nn.compact
    def __call__(self, x: Array) -> Array:
        red = self.red(x)
        green = self.green(x)
        blue = self.blue(x)
        return jnp.stack((red, green, blue), axis=jnp.ndim(red) - 1)
