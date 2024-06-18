from typing import Optional, List

import flax.linen as nn
from flax.training.common_utils import onehot

from src.enviroment import Shape
from src.models.dreamer.variationalencoder import VariationalEncoder
from src.models.helpers import linear_layer_init
from src.utils.activationfuns import activation_function_dict
import jax
import jax.numpy as jnp

from src.utils.modelhelperfuns import sample_normal


class TransitionModel(nn.Module):
    belief_size: int
    state_size: int
    action_size: int
    hidden_size: int

    def setup(self):
        super().__init__()
        self.activation_fun = activation_function_dict['relu']
        self.rnn = nn.GRUCell(features=self.belief_size)
        self.variational_encoder = VariationalEncoder(2 * self.state_size)
        self.embedding_size = 100
        self.belief_dense = linear_layer_init(self.belief_size)

    def update_belief(self, belief, state, action):
        state_action = jnp.append(state, action, axis=-1)
        state_action = self.belief_dense(state_action)
        state_action = self.activation_fun(state_action)
        belief, _ = self.rnn(belief, state_action)
        return belief

    def prior_update(self, belief):
        # Compute state prior by applying transition dynamics
        hidden = linear_layer_init(self.hidden_size)(belief)
        hidden = self.activation_fun(hidden)
        return self.variational_encoder(hidden)

    @nn.compact
    def __call__(self, prev_state: jax.Array, actions: jax.Array, prev_belief: jax.Array) -> List[jax.Array]:
        actions = onehot(actions, Shape()[1])
        beliefs = self.update_belief(prev_belief, prev_state, actions)
        state, std_dev, mean = self.prior_update(beliefs)
        return beliefs, state, std_dev, mean
