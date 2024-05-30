import flax.linen as nn
import jax
import jax.numpy as jnp

from src.singletons.hyperparameters import Args
from src.utils.modelhelperfuns import sample_normal


class VariationalEncoder(nn.Module):
    hidden_size: int

    def setup(self) -> None:
        self.min_std_dev: float = Args().args.min_std_dev

    @nn.compact
    def __call__(self, data):
        hidden = nn.Dense(self.hidden_size)(data)
        mean, std_dev = jnp.array_split(hidden, 2, axis=1)
        std_dev = nn.softplus(std_dev) + self.min_std_dev
        jax.debug.print("mean {m}, std_dev {s}", m=mean, s=std_dev)
        sample = sample_normal(self.make_rng('normal'), mean, std_dev)
        return sample, mean, std_dev