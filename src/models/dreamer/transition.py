from typing import Optional, List

import flax.linen as nn
from flax.training.common_utils import onehot

from src.enviroment import Shape
from src.models.dreamer.variationalencoder import VariationalEncoder
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
        self.belief_dense = nn.Dense(self.state_size + self.action_size, self.belief_size)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # d
    def update_belief(self, belief, state, action):
        state_action = jnp.append(state, action, axis=-1)
        state_action = self.belief_dense(state_action)
        state_action = self.activation_fun(state_action)
        belief, _ = self.rnn(belief, state_action)
        return belief

    def prior_update(self, belief):
        # Compute state prior by applying transition dynamics
        hidden = nn.Dense(self.belief_size, self.hidden_size)(belief)
        hidden = self.activation_fun(hidden)
        return self.variational_encoder(hidden)

    @nn.compact
    def __call__(self, prev_state: jax.Array, actions: jax.Array, prev_belief: jax.Array,
                 nonterminals: Optional[jax.Array] = None) \
            -> List[jax.Array]:
        """
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        actions = onehot(actions, Shape()[1])
        beliefs = self.update_belief(prev_belief, prev_state, actions)
        state, std_dev, mean = self.prior_update(beliefs)
        return beliefs, state, std_dev, mean
