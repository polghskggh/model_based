from flax import linen as nn
import jax.numpy as jnp

from src.models.helpers import linear_layer_init
from src.singletons.hyperparameters import Args
from src.utils.activationfuns import activation_function_dict


class PredictModel(nn.Module):
    embedding_size: int
    outputs: int
    activation_function: str = 'relu'

    def setup(self):
        self.hidden_size = 500
        self.activation_fun = activation_function_dict[self.activation_function]

    @nn.compact
    def __call__(self, belief, state):
        hidden = jnp.append(belief, state, axis=1)

        for _ in range(3):
            hidden = linear_layer_init(features=self.hidden_size)(hidden)
            hidden = self.activation_fun(hidden)

        output = linear_layer_init(features=self.outputs)(hidden)
        return output

