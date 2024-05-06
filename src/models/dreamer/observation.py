import flax.linen as nn
import jax.numpy as jnp

from src.utils.activationfuns import activation_function_dict


class ObservationModel(nn.Module):
    belief_size: int
    state_size: int
    embedding_size: int
    activation_function: str = 'relu'

    def setup(self):
        self.activation_fun = activation_function_dict[self.activation_function]
        self.embedding_size = self.embedding_size
        self.dense_layer = nn.Dense(self.belief_size + self.state_size, self.embedding_size)
        self.conv1 = nn.ConvTranspose(self.embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose(32, 3, 6, stride=2)

    def __call__(self, belief, state):
        hidden = self.dense_layer(jnp.append(belief, state, axis=1))  # No nonlinearity here
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)

        for layer in [self.conv1, self.conv2, self.conv3]:
            hidden = self.activation_fun(layer(hidden))

        observation = self.conv4(hidden)
        return observation

