from typing import Optional, List

import flax.linen as nn

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
    activation_function: str = 'relu'

    def setup(self):
        super().__init__()
        self.activation_fun = activation_function_dict[self.activation_function]
        self.rnn = nn.GRUCell(features=self.belief_size)
        self.variational_encoder = VariationalEncoder(2 * self.state_size)
        self.embedding_size = 100

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
        # Compute belief (deterministic hidden state)

        print(state.shape, action.shape)
        state_action = jnp.append(state, action, axis=-1)
        state_action = nn.Dense(self.state_size + self.action_size, self.belief_size)(state_action)
        state_action = self.activation_fun(state_action)
        belief = self.rnn(belief, state_action)
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
        batch = actions.shape[0] + 1
        beliefs = [jnp.empty(0)] * batch
        # TODO: incorrect dims
        prior_states = [jnp.empty(0)] * batch
        prior_means = [jnp.empty(0)] * batch
        prior_std_devs = [jnp.empty(0)] * batch

        beliefs[0], prior_states[0] = prev_belief, prev_state
        # Loop over time sequence
        print(beliefs[0].shape, actions[0].shape, prior_states[0].shape)
        for t in range(batch - 1):
            # Mask if previous transition was terminal
            state = prior_states[t]
            state = state if (nonterminals is None or t == 0) else state * nonterminals[t - 1]
            beliefs[t + 1] = self.update_belief(beliefs[t], state, actions[t])
            prior_std_devs[t + 1], prior_means[t + 1], prior_std_devs[t + 1] = self.prior_update(beliefs[t + 1])

        # Return new hidden states
        hidden = [jnp.stack(tuple(beliefs[1:])), jnp.stack(tuple(prior_states[1:])),
                  jnp.stack(tuple(prior_means[1:])), jnp.stack(tuple(prior_std_devs[1:]))]

        return hidden
