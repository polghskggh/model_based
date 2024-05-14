import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp


from src.models.base.cnnatari import CNNAtari


class ActorAtari(nn.Module):
    input_dimensions: tuple
    output_dimensions: int
    deterministic: bool = True

    def setup(self):
        self.cnn = CNNAtari(100, deterministic=self.deterministic)

    @nn.compact
    def __call__(self, x: Array):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.softmax(x)
        return x

    def deterministic(self, deterministic: bool):
        self.cnn.encoder.deterministic = deterministic
