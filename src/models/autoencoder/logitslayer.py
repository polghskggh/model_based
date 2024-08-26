from flax import linen as nn
from jax import Array, lax
import jax.numpy as jnp

from src.models.helpers import linear_layer_init
from src.singletons.hyperparameters import Args


# logits
class LogitsLayer(nn.Module):
    def setup(self):
        self.features = 256
        self.red = linear_layer_init(features=self.features)
        self.green = linear_layer_init(features=self.features)
        self.blue = linear_layer_init(features=self.features)
        self.grayscale = linear_layer_init(features=self.features)

    def rgb(self, x: Array) -> Array:
        return jnp.stack((self.red(x), self.green(x), self.blue(x)), axis=jnp.ndim(x) - 1)

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if Args().args.grayscale:
            return self.grayscale(x)
        else:
            return self.rgb(x)
