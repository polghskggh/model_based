from typing import Optional, List

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.utils.activationfuns import activation_function_dict
from src.utils.modelhelperfuns import sample_normal


def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Dense(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Dense(belief_size, hidden_size)
        self.fc_state_prior = nn.Dense(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Dense(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Dense(hidden_size, 2 * state_size)
        self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior,
                        self.fc_embed_belief_posterior, self.fc_state_posterior]

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
    def __call__(self, prev_state: jax.Array, actions: jax.Array, prev_belief: jax.Array,
            observations: Optional[jnp.ndarray] = None, nonterminals: Optional[jax.Array] = None)\
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
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]

            # Mask if previous transition was terminal
            _state = _state if (nonterminals is None or t == 0) else _state * nonterminals[t - 1]

            # Compute belief (deterministic hidden state)
            state_action = self.activation_fun(self.fc_embed_state_action(jnp.append(_state, actions[t])))
            beliefs[t + 1] = self.rnn(state_action, beliefs[t])

            # Compute state prior by applying transition dynamics
            hidden = self.activation_fun(self.fc_embed_belief_prior(beliefs[t + 1]))

            prior_means[t + 1], _prior_std_dev = jnp.array_split(self.fc_state_prior(hidden), 2, axis=1)
            prior_std_devs[t + 1] = nn.softplus(_prior_std_dev) + self.min_std_dev
            
            prior_states[t + 1] = sample_normal(self.make_rng('normal'), prior_means[t + 1], prior_std_devs[t + 1])

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.activation_fun(
                    self.fc_embed_belief_posterior(jnp.append(beliefs[t + 1], observations[t_ + 1], axis=1)))

                posterior_means[t + 1], _posterior_std_dev = jnp.array_split(self.fc_state_posterior(hidden), 2, axis=1)
                posterior_std_devs[t + 1] = nn.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = sample_normal(self.make_rng('normal'), posterior_means[t + 1],
                                                        posterior_std_devs[t + 1])

        # Return new hidden states
        hidden = [jnp.stack(tuple(beliefs[1:])), jnp.stack(tuple(prior_states[1:])),
                  jnp.stack(tuple(prior_means[1:])), jnp.stack(tuple(prior_std_devs[1:]))]
        if observations is not None:
            hidden += [jnp.stack(tuple(posterior_states[1:])), jnp.stack(tuple(posterior_means[1:])),
                       jnp.stack(tuple(posterior_std_devs[1:]))]
        return hidden


class SymbolicObservationModel(nn.Module):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.fc1 = nn.Dense(belief_size + state_size, embedding_size)
        self.fc2 = nn.Dense(embedding_size, embedding_size)
        self.fc3 = nn.Dense(embedding_size, observation_size)

    def forward(self, belief, state):
        hidden = self.activation_fun(self.fc1(jnp.append(belief, state, axis=1)))
        hidden = self.activation_fun(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(nn.Module):
    def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.embedding_size = embedding_size
        self.fc1 = nn.Dense(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose(32, 3, 6, stride=2)

    def forward(self, belief, state):
        hidden = self.fc1(jnp.append(belief, state, axis=1))  # No nonlinearity here
        hidden = hidden.reshape(-1, self.embedding_size, 1, 1)
        hidden = self.activation_fun(self.conv1(hidden))
        hidden = self.activation_fun(self.conv2(hidden))
        hidden = self.activation_fun(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.fc1 = nn.Dense(belief_size + state_size, hidden_size)
        self.fc2 = nn.Dense(hidden_size, hidden_size)
        self.fc3 = nn.Dense(hidden_size, 1)

    def __call__(self, belief, state):
        x = jnp.append(belief, state, axis=1)
        hidden = self.activation_fun(self.fc1(x))
        hidden = self.activation_fun(self.fc2(hidden))
        reward = self.fc3(hidden)
        reward = jnp.squeeze(reward, axis=-1)
        return reward


class ValueModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.fc1 = nn.Dense(belief_size + state_size, hidden_size)
        self.fc2 = nn.Dense(hidden_size, hidden_size)
        self.fc3 = nn.Dense(hidden_size, hidden_size)
        self.fc4 = nn.Dense(hidden_size, 1)

    def __call__(self, belief, state):
        x = jnp.append(belief, state, axis=1)
        hidden = self.activation_fun(self.fc1(x))
        hidden = self.activation_fun(self.fc2(hidden))
        hidden = self.activation_fun(self.fc3(hidden))
        reward = jnp.squeeze(self.fc4(hidden), axis=1)
        return reward


class SymbolicEncoder(nn.Module):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.fc1 = nn.Dense(observation_size, embedding_size)
        self.fc2 = nn.Dense(embedding_size, embedding_size)
        self.fc3 = nn.Dense(embedding_size, embedding_size)

    def __call__(self, observation):
        hidden = self.activation_fun(self.fc1(observation))
        hidden = self.activation_fun(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv(3, 32, 4, stride=2)
        self.conv2 = nn.Conv(32, 64, 4, stride=2)
        self.conv3 = nn.Conv(64, 128, 4, stride=2)
        self.conv4 = nn.Conv(128, 256, 4, stride=2)
        self.fc = lambda x: x if embedding_size == 1024 else nn.Dense(1024, embedding_size)

    def __call__(self, observation):
        hidden = self.activation_fun(self.conv1(observation))
        hidden = self.activation_fun(self.conv2(hidden))
        hidden = self.activation_fun(self.conv3(hidden))
        hidden = self.activation_fun(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)


class PCONTModel(nn.Module):
    """ predict the prob of whether a state is a terminal state. """

    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.fc1 = nn.Dense(belief_size + state_size, hidden_size)
        self.fc2 = nn.Dense(hidden_size, hidden_size)
        self.fc3 = nn.Dense(hidden_size, hidden_size)
        self.fc4 = nn.Dense(hidden_size, 1)

    def __call__(self, belief, state):
        x = jnp.append(belief, state, axis=1)
        hidden = self.activation_fun(self.fc1(x))
        hidden = self.activation_fun(self.fc2(hidden))
        hidden = self.activation_fun(self.fc3(hidden))
        x = jnp.squeeze(self.fc4(hidden), axis=1)
        p = nn.sigmoid(x)
        return p

#
# class ActorModel(nn.Module):
#     def __init__(self, action_size, belief_size, state_size, hidden_size, mean_scale=5, min_std=1e-4, init_std=5,
#                  activation_function="elu"):
#         super().__init__()
#         self.activation_fun = getattr(F, activation_function)
#         self.fc1 = nn.Dense(belief_size + state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Dense(hidden_size, hidden_size)
#         self.fc4 = nn.Linear(hidden_size, hidden_size)
#         self.fc5 = nn.Linear(hidden_size, 2 * action_size)
#         self.min_std = min_std
#         self.init_std = init_std
#         self.mean_scale = mean_scale
#
#     def forward(self, belief, state, deterministic=False, with_logprob=False):
#         raw_init_std = np.log(np.exp(self.init_std) - 1)
#         hidden = self.activation_fun(self.fc1(torch.cat([belief, state], dim=-1)))
#         hidden = self.activation_fun(self.fc2(hidden))
#         hidden = self.activation_fun(self.fc3(hidden))
#         hidden = self.activation_fun(self.fc4(hidden))
#         hidden = self.fc5(hidden)
#         mean, std = torch.chunk(hidden, 2, dim=-1)
#         mean = self.mean_scale * torch.tanh(
#             mean / self.mean_scale)  # bound the action to [-5, 5] --> to avoid numerical instabilities.  For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
#         std = F.softplus(std + raw_init_std) + self.min_std
#         dist = torch.distributions.Normal(mean, std)
#         transform = [torch.distributions.transforms.TanhTransform()]
#         dist = torch.distributions.TransformedDistribution(dist, transform)
#         dist = torch.distributions.independent.Independent(dist, 1)  # Introduces dependence between actions dimension
#         dist = SampleDist(
#             dist)  # because after transform a distribution, some methods may become invalid, such as entropy, mean and mode, we need SmapleDist to approximate it.
#
#         if deterministic:
#             action = dist.mean
#         else:
#             action = dist.rsample()
#
#         if with_logprob:
#             logp_pi = dist.log_prob(action)
#         else:
#             logp_pi = None
#
#         return action, logp_pi