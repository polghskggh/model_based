import flax.linen as nn
import jax

from src.models.base.cnnatari import CNNAtari
from src.models.helpers import linear_layer_init


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
        self.bottleneck = 200

    @nn.compact
    def __call__(self, state: jax.Array):
        hidden = linear_layer_init(self.bottleneck)(state)
        hidden = nn.relu(hidden)
        hidden = linear_layer_init(self.bottleneck)(hidden)
        hidden = nn.relu(hidden)
        hidden = linear_layer_init(self.bottleneck)(hidden)
        hidden = nn.relu(hidden)
        policy = linear_layer_init(self.output_dimensions[0])(hidden)
        value = linear_layer_init(self.output_dimensions[1])(hidden)
        return policy, value
