from ctypes import Array

import flax.linen as nn
import jax.lax

import jax.numpy as jnp

from src.models.helpers import linear_layer_init
from src.singletons.hyperparameters import Args


class Predictor(nn.Module):
    outputs: int

    @nn.compact
    def __call__(self, middle: Array, final: Array) -> Array:
        middle = middle.reshape(middle.shape[0], -1)
        final = jax.lax.reduce(final, 0.0, lambda x, y: x + y, (1, 2))
        discount_pred = jnp.concat((middle, final), axis=-1)
        discount_pred = linear_layer_init(128)(discount_pred)
        discount_pred = nn.relu(discount_pred)
        discount_pred = linear_layer_init(self.outputs)(discount_pred)
        return discount_pred
