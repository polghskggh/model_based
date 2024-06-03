from flax import linen as nn
from jax import Array, lax
import jax.numpy as jnp
from src.singletons.hyperparameters import Args


# logits
class LogitsLayer(nn.Module):
    def setup(self):
        self.features = 256
        self.red = nn.Dense(features=self.features, name="red_logits")
        self.green = nn.Dense(features=self.features, name="greed_logits")
        self.blue = nn.Dense(features=self.features, name="blue_logits")
        self.grayscale = nn.Dense(features=self.features, name="grayscale_logits")
        self.type = Args().args.grayscale

    def rgb(self, x: Array) -> Array:
        return jnp.stack((self.red(x), self.green(x), self.blue(x)), axis=jnp.ndim(x) - 1)

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return lax.cond(self.type, self.grayscale, self.rgb, x)
