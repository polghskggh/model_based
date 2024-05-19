import flax.linen as nn
import jax.numpy as jnp
from jax import Array
from rlax import one_hot

from src.models.agent.actoratari import CNNAtari
from src.models.base.mlpatari import MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    second_input: int
    output_dimensions: int
    deterministic: bool = True

    def setup(self):
        bottleneck = 100
        self.cnn = CNNAtari(bottleneck)
        self.mlp = MLPAtari(bottleneck + self.second_input, self.output_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
        print(action.shape)
        action = one_hot(action, self.second_input)
        print(action.shape)
        x = jnp.append(cnn, action, axis=-1)
        x = self.mlp(x)
        return x


class StateValueAtariNN(nn.Module):
    input_dimensions: tuple
    output_dimensions: int
    deterministic: bool = True

    def setup(self):
        bottleneck = 100
        self.cnn = CNNAtari(bottleneck, self.deterministic)

    @nn.compact
    def __call__(self, image: Array):
        cnn = self.cnn(image)
        x = nn.Dense(self.output_dimensions)(cnn)
        return x

