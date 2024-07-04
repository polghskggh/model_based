import flax.linen as nn
import jax
import jax.numpy as jnp

from src.models.autoencoder.decoder import Decoder
from src.models.autoencoder.logitslayer import LogitsLayer
from src.models.helpers import linear_layer_init
from src.singletons.hyperparameters import Args
from src.utils.activationfuns import activation_function_dict


class ObservationModel(nn.Module):
    belief_size: int
    state_size: int
    embedding_size: int
    output_dims: tuple
    activation_function: str = 'relu'

    def setup(self):
        self.activation_fun = activation_function_dict[self.activation_function]
        self.decoder = Decoder(Args().args.bottleneck_dims[-1], deterministic=True, normalization=False)

    @nn.compact
    def __call__(self, belief, state):
        hidden = jnp.append(belief, state, axis=-1)
        hidden = linear_layer_init(features=self.embedding_size)(hidden)
        hidden = hidden.reshape(-1, *Args().args.bottleneck_dims)
        reconstructed = self.decoder(hidden, None)
        reconstructed = LogitsLayer()(reconstructed)
        return reconstructed
