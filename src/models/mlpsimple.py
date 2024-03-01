from flax import linen as nn
from jax import Array


# A simple feed forward neural network
class MLPSimple(nn.Module):
    input_dimensions: int
    output_dimensions: int

    @nn.compact
    def __call__(self, x) -> Array:
        x = nn.Dense(2 * self.input_dimensions)(x)
        x = nn.relu(x)
        x = nn.Dense(self.input_dimensions)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        return x
