import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from src.models.helpers import convolution_layer_init, linear_layer_init


class Network(nn.Module):
    input_dimensions: tuple
    output_dimensions: tuple

    @nn.compact
    def __call__(self, x: Array):
        x = nn.Sequential([
            convolution_layer_init(32, 8, 4),
            nn.relu,
            convolution_layer_init(64, 4, 2),
            nn.relu,
            convolution_layer_init(64, 3, 1),
            nn.relu,
        ])(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Sequential([
            linear_layer_init(512),
            nn.relu
        ])(x)
        actor = Actor(4)(x)
        critic = Critic()(x)
        return actor, critic


class Actor(nn.Module):
    action_n: int

    @nn.compact
    def __call__(self, x: Array):
        return linear_layer_init(self.action_n, std=0.01)(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array):
        return linear_layer_init(1, std=1)(x)




