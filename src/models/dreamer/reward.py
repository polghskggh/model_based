from flax import linen as nn
import jax.numpy as jnp

from src.utils.activationfuns import activation_function_dict


class RewardModel(nn.Module):
    belief_size: int
    state_size: int
    embedding_size: int
    activation_function: str = 'relu'

    def setup(self):
        self.activation_fun =  activation_function_dict[self.activation_function]
        self.fc1 = nn.Dense(self.belief_size + self.state_size, self.hidden_size)
        self.fc2 = nn.Dense(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Dense(self.hidden_size, 1)

    def __call__(self, belief, state):
        x = jnp.append(belief, state, axis=1)
        hidden = self.activation_fun(self.fc1(x))
        hidden = self.activation_fun(self.fc2(hidden))
        reward = self.fc3(hidden)
        reward = jnp.squeeze(reward, axis=-1)
        return reward

