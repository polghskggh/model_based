import flax.linen as nn
import jax.numpy as jnp

from src.utils.activationfuns import activation_function_dict


class ObservationModel(nn.Module):
    belief_size: int
    state_size: int
    embedding_size: int
    output_dims: tuple
    activation_function: str = 'relu'

    def setup(self):
        self.activation_fun = activation_function_dict[self.activation_function]
        self.dense_layer = nn.Dense(self.belief_size + self.state_size, self.embedding_size)
        self.layers = 4
        self.features = 256

    def __call__(self, belief, state):
        hidden = self.dense_layer(jnp.append(belief, state, axis=1))  # No nonlinearity here
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)

        for idx in range(6):
            features = self.scaled_features(idx)
            hidden = nn.ConvTranspose(hidden, features=features, kernel_size=(3, 3), strides=(2, 2))
            hidden = self.activation_fun(hidden)

        return hidden

    def scaled_features(self, layer_id: int):
        if layer_id < 4:
            return self.features
        if layer_id == 4:
            return self.features // 2
        if layer_id == 5:
            return self.features // 4

