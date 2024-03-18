from flax import linen as nn
from jax import Array


# encoder
class Encoder(nn.Module):
    input_dimensions: tuple

    @nn.compact
    def __call__(self, x) -> Array:
        x = nn.avg_pool(x, window_shape=(4, 4))
        x = nn.Conv(features=9, kernel_size=(4, 4), strides=4)(x)
        x = nn.avg_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=9, kernel_size=(4, 4), strides=4)(x)
        x = nn.avg_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=9, kernel_size=(4, 4), strides=4)(x)
        return x
