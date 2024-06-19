from typing import List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.common_utils import onehot

from src.enviroment import Shape
from src.models.autoencoder.encoder import Encoder
from src.models.dreamer.transition import TransitionModel
from src.models.dreamer.variationalencoder import VariationalEncoder
from src.models.helpers import linear_layer_init
from src.utils.activationfuns import activation_function_dict
from src.utils.modelhelperfuns import sample_normal


class RepresentationModel(nn.Module):
    belief_size: int
    state_size: int
    action_size: int
    hidden_size: int
    embedding_size: int
    observation_shape: tuple
    activation_function: str = 'relu'
    min_std_dev: float = 0.1
    __constants__ = ['min_std_dev']

    def setup(self):
        super().__init__()
        self.activation_fun = activation_function_dict[self.activation_function]
        self.rnn = nn.GRUCell(self.belief_size, self.belief_size)
        self.variational_encoder = VariationalEncoder(2 * self.state_size)
        self.transition_model = TransitionModel(self.belief_size, self.state_size, self.hidden_size)
        self.posterior_dense = linear_layer_init(self.hidden_size)
        self.encoder = Encoder(64)

    def posterior_update(self, belief, encoded_observation):
        # Compute state posterior by applying transition dynamics and using current observation
        encoded_observation = encoded_observation.reshape(encoded_observation.shape[0], -1)
        x = jnp.append(belief, encoded_observation, axis=-1)
        hidden = self.posterior_dense(x)
        hidden = self.activation_fun(hidden)

        return self.variational_encoder(hidden)

    def __call__(self, prev_state: jax.Array, actions: jax.Array, prev_belief: jax.Array,
                 encoded_observations: jax.Array):
        beliefs, _, prior_means, prior_std_devs = self.transition_model(prev_state, actions, prev_belief)
        posterior_states, posterior_means, posterior_std_devs = self.posterior_update(beliefs, encoded_observations)
        return beliefs, posterior_states, prior_means, prior_std_devs, posterior_means, posterior_std_devs

