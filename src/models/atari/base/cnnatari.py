from flax import linen as nn
from jax import Array


# A simple feed forward neural network
class CNNAtari(nn.Module):
    input_dimensions: tuple
    output_dimensions: int

    @nn.compact
    def __call__(self, x) -> Array:
        x = nn.avg_pool(x, window_shape=(4, 4))
        x = nn.Conv(features=9, kernel_size=(3, 3))(x)
        x = nn.avg_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=9, kernel_size=(3, 3))(x)
        x = nn.avg_pool(x, window_shape=(2, 2))
        x = self.flatten(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.relu(x)
        return x

    @staticmethod
    def flatten(x: Array) -> Array:
        return x.reshape(-1) if x.ndim == 3 else x.reshape((x.shape[0], -1))
