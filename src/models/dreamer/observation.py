import flax.linen as nn
import jax.numpy as jnp

from src.models.autoencoder.decoder import Decoder
from src.models.autoencoder.logitslayer import LogitsLayer
from src.models.helpers import linear_layer_init
from src.utils.activationfuns import activation_function_dict


class ObservationModel(nn.Module):
    belief_size: int
    state_size: int
    embedding_size: int
    output_dims: tuple
    activation_function: str = 'relu'

    def setup(self):
        self.activation_fun = activation_function_dict[self.activation_function]
        self.layers = 4
        self.features = 256
        self.decoder = Decoder(64)

    @nn.compact
    def __call__(self, belief, state):
        hidden = jnp.append(belief, state, axis=1)
        hidden = linear_layer_init(features=self.embedding_size)(hidden)
        hidden = hidden.reshape(-1, 2, 2, self.embedding_size // 4)

        reconstructed = self.decoder(hidden, None)
        reconstructed = LogitsLayer()(reconstructed)
        return reconstructed

    def scaled_features(self, layer_id: int):
        if layer_id < 4:
            return self.features
        if layer_id == 4:
            return self.features // 2
        if layer_id == 5:
            return self.features // 4

