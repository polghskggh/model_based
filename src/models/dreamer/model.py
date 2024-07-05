import flax.linen as nn
import jax.numpy as jnp

from src.utils.activationfuns import activation_function_dict


def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output


class ValueModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.activation_fun = activation_function_dict[activation_function]
        self.fc1 = nn.Dense(belief_size + state_size, hidden_size)
        self.fc2 = nn.Dense(hidden_size, hidden_size)
        self.fc3 = nn.Dense(hidden_size, hidden_size)
        self.fc4 = nn.Dense(hidden_size, 1)

    def __call__(self, belief, state):
        x = jnp.append(belief, state, axis=-1)
        hidden = self.activation_fun(self.fc1(x))
        hidden = self.activation_fun(self.fc2(hidden))
        hidden = self.activation_fun(self.fc3(hidden))
        reward = jnp.squeeze(self.fc4(hidden), axis=-1)
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
        x = jnp.append(belief, state, axis=-1)
        hidden = self.activation_fun(self.fc1(x))
        hidden = self.activation_fun(self.fc2(hidden))
        hidden = self.activation_fun(self.fc3(hidden))
        x = jnp.squeeze(self.fc4(hidden), axis=-1)
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