import torch
import torch.nn as nn
from torch.distributions import TanhTransform

from src.altdreamer.utils import build_network, create_normal_dist


class Actor(nn.Module):
    def __init__(self, discrete_action_bool, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        action_size = action_size if discrete_action_bool else 2 * action_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            action_size,
        )

    def forward(self, posterior, deterministic):
        x = torch.cat((posterior, deterministic), -1)
        x = self.network(x)
        dist = torch.distributions.OneHotCategorical(logits=x)
        action = dist.sample() + dist.probs - dist.probs.detach()
        return action