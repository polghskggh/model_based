import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.helpers import convolution_layer_init, linear_layer_init


class AtariNN(nn.Module):
    input_dimensions: tuple
    output_dimensions: tuple

    @nn.compact
    def __call__(self, x: Array):
        x = convolution_layer_init(32, 8, 4)(x)
        x = nn.relu(x)
        x = convolution_layer_init(64, 4, 2)(x)
        x = nn.relu(x)
        x = convolution_layer_init(64, 3, 1)(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = linear_layer_init(512)(x)
        x = nn.relu(x)
        x = linear_layer_init(1, std=1)(x)
        return x
