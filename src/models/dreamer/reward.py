from flax import linen as nn
import jax.numpy as jnp

from src.utils.activationfuns import activation_function_dict


class RewardModel(nn.Module):
    belief_size: int
    state_size: int
    embedding_size: int
    activation_function: str = 'relu'

    def setup(self):
        self.hidden_size = 256
        self.activation_fun = activation_function_dict[self.activation_function]

    def __call__(self, belief, state):
        x = jnp.append(belief, state, axis=1)

        hidden = nn.Dense(features=self.hidden_size)(x)
        hidden = self.activation_fun(hidden)

        hidden = nn.Dense(features=self.hidden_size)(hidden)
        hidden = self.activation_fun(hidden)

        output = nn.Dense(features=1)(hidden)
        reward = jnp.squeeze(output, axis=-1)
        return reward

