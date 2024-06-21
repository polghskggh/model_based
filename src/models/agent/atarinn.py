import flax.linen as nn
import jax.numpy as jnp
from jax import Array
from rlax import one_hot

from src.models.helpers import convolution_layer_init, linear_layer_init


class AtariNN(nn.Module):
    input_dimensions: tuple
    action_n: int
    output_dimensions: tuple

    @nn.compact
    def __call__(self, x: Array, action: int):
        x = convolution_layer_init(32, 8, 4)(x)
        x = nn.relu(x)
        x = convolution_layer_init(64, 4, 2)(x)
        x = nn.relu(x)
        x = convolution_layer_init(64, 3, 1)(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = linear_layer_init(512)(x)
        x = nn.relu(x)

        action = one_hot(action, self.action_n)
        x = jnp.append(x, action, axis=-1)

        x = linear_layer_init(300, std=1)(x)
        x = nn.relu(x)
        x = linear_layer_init(100, std=1)(x)
        x = nn.relu(x)
        x = linear_layer_init(1, std=1)(x)
        return x
