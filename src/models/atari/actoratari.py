import flax.linen as nn
from jax import Array

from src.models.atari.cnnatari import CNNAtari


class ActorAtari(nn.Module):
    input_dimensions: tuple
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions[0], self.input_dimensions[1], self.input_dimensions[2], 10)

    @nn.compact
    def __call__(self, x: Array):
        x = self.cnn(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimensions)(x)
        x = nn.softmax(x)
        return x



