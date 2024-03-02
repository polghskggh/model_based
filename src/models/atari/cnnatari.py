from flax import linen as nn
from jax import Array
import jax.numpy as jnp


# A simple feed forward neural network
class CNNAtari(nn.Module):
    x: int
    y: int
    channels: int
    output_dimensions: int

    @nn.compact
    def __call__(self, x) -> Array:
        x = nn.avg_pool(x, window_shape=(4, 4))
        x = nn.Conv(features=self.channels * 3, kernel_size=(3, 3))(x)
        x = nn.avg_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=self.channels * 3, kernel_size=(3, 3))(x)
        x = nn.avg_pool(x, window_shape=(2, 2))
        x = x.reshape(-1, jnp.prod(x.shape[1:]))  # Flatten
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.relu(x)
        return x
