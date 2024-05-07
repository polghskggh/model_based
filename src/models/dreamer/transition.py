from typing import Optional, List

import flax.linen as nn

from src.utils.activationfuns import activation_function_dict
import jax
import jax.numpy as jnp

from src.utils.modelhelperfuns import sample_normal


class TransitionModel(nn.Module):
    belief_size: int
    state_size: int
    action_size: int
    hidden_size: int
    embedding_size: int
    activation_function: str = 'relu'
    min_std_dev: float = 0.1
    __constants__ = ['min_std_dev']

    def setup(self):
        super().__init__()
        self.activation_fun = activation_function_dict[self.activation_function]
        self.dense_embed_state_action = nn.Dense(self.state_size + self.action_size, self.belief_size)
        self.rnn = nn.GRUCell(self.belief_size, self.belief_size)
        self.dense_embed_belief_prior = nn.Dense(self.belief_size, self.hidden_size)
        self.dense_state_prior = nn.Dense(self.hidden_size, 2 * self.state_size)
        self.dense_embed_belief_posterior = nn.Dense(self.belief_size + self.embedding_size, self.hidden_size)
        self.dense_state_posterior = nn.Dense(self.hidden_size, 2 * self.state_size)
        self.modules = [self.dense_embed_state_action, self.dense_embed_belief_prior, self.dense_state_prior,
                        self.dense_embed_belief_posterior, self.dense_state_posterior]

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
    # @jit.script_method

    def variational_encode(self, data):
        mean, std_dev = jnp.array_split(self.dense_state_posterior(data), 2, axis=1)
        std_dev = nn.softplus(std_dev) + self.min_std_dev
        sample = sample_normal(self.make_rng('normal'), mean, std_dev)
        return sample, mean, std_dev

    def update_belief(self, state, belief, action):
        # Compute belief (deterministic hidden state)
        state_action = self.activation_fun(self.dense_embed_state_action(jnp.append(state, action)))
        belief = self.rnn(state_action, belief)
        return belief

    def posterior_update(self, belief, observation):
        # Compute state posterior by applying transition dynamics and using current observation
        hidden = self.activation_fun(
            self.dense_embed_belief_posterior(jnp.append(belief, observation, axis=0)))
        return self.variational_encode(hidden)

    def prior_update(self, belief):
        # Compute state prior by applying transition dynamics
        hidden = self.activation_fun(self.dense_embed_belief_prior(belief))
        return self.variational_encode(hidden)

    def __call__(self, prev_state: jax.Array, actions: jax.Array, prev_belief: jax.Array,
                 observations: Optional[jnp.ndarray] = None, nonterminals: Optional[jax.Array] = None) \
            -> List[jax.Array]:
        """
        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        """
        batch = actions.shape[0] + 1
        beliefs = [jnp.empty(0)] * batch
        prior_states = [jnp.empty(0)] * batch
        prior_means = [jnp.empty(0)] * batch
        prior_std_devs = [jnp.empty(0)] * batch
        posterior_states = [jnp.empty(0)] * batch
        posterior_means = [jnp.empty(0)] * batch
        posterior_std_devs = [jnp.empty(0)] * batch

        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        # Loop over time sequence
        for t in range(batch - 1):
            # Mask if previous transition was terminal
            state = prior_states[t] if observations is None else posterior_states[t]
            state = state if (nonterminals is None or t == 0) else state * nonterminals[t - 1]
            beliefs[t + 1] = self.update_belief(beliefs[t], state, actions[t])
            prior_std_devs[t + 1], prior_means[t + 1], prior_std_devs[t + 1] = self.time_step_update(beliefs[t + 1])

            if observations is not None:
                posterior_states[t + 1], posterior_means[t + 1], posterior_std_devs[t + 1] = self.posterior_update(
                    beliefs[t + 1], observations[t])

        # Return new hidden states
        hidden = [jnp.stack(tuple(beliefs[1:])), jnp.stack(tuple(prior_states[1:])),
                  jnp.stack(tuple(prior_means[1:])), jnp.stack(tuple(prior_std_devs[1:]))]

        if observations is not None:
            hidden += [jnp.stack(tuple(posterior_states[1:])), jnp.stack(tuple(posterior_means[1:])),
                       jnp.stack(tuple(posterior_std_devs[1:]))]

        return hidden
