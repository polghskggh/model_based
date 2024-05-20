import flax.linen as nn
import jax

from src.models.base.cnnatari import CNNAtari


class ActorCritic(nn.Module):
    input_dimensions: tuple
    output_dimensions: tuple
    deterministic: bool = True

    def setup(self):
        bottleneck = 100
        self.cnn = CNNAtari(bottleneck, self.deterministic)

    @nn.compact
    def __call__(self, image: jax.Array):
        hidden = self.cnn(image)
        policy = nn.Dense(self.output_dimensions[0])(hidden)
        value = nn.Dense(self.output_dimensions[1])(hidden)
        return policy, value


class ActorCriticDreamer(nn.Module):
    input_dimensions: tuple
    output_dimensions: tuple
    deterministic: bool = True

    def setup(self):
        self.bottleneck = 100

    @nn.compact
    def __call__(self, state: jax.Array):
        hidden = nn.Dense(self.bottleneck)(state)
        policy = nn.Dense(self.output_dimensions[0])(hidden)
        value = nn.Dense(self.output_dimensions[1])(hidden)
        return policy, value
