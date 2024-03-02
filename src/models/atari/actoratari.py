import flax.linen as nn
from jax import Array

from src.models.atari import CNNAtari


class ActorAtari(nn.Module):
    input_dimensions: tuple
    output_dimensions: int

    def setup(self):
        self.cnn = CNNAtari(self.input_dimensions[0], self.input_dimensions[1], self.input_dimensions[2], 10)

    def __call__(self, x: Array):
        x = self.cnn(x)
        x = nn.softmax(x)
        return x



