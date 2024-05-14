import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from src.models.actorcritic.actoratari import CNNAtari
from src.models.base.mlpatari import MLPAtari


class AtariNN(nn.Module):
    input_dimensions: tuple
    second_input: int
    output_dimensions: int
    deterministic: bool = True

    def setup(self):
        bottleneck = 1000
        self.cnn = CNNAtari(bottleneck)
        self.mlp = MLPAtari(bottleneck + self.second_input, self.output_dimensions)

    @nn.compact
    def __call__(self, image: Array, action: Array):
        cnn = self.cnn(image)
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


    def deterministic(self, deterministic: bool):
        self.cnn.encoder.deterministic = deterministic
