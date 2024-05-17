import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class PPOStorage:
    observations: jnp.array
    rewards: jnp.array
    actions: jnp.array
    log_probs: jnp.array
    dones: jnp.array


@dataclass
class DQNStorage:
    observations: jnp.array
    rewards: jnp.array
    actions: jnp.array
    next_observations: jnp.array


def store(storage, step: slice | int, **kwargs):
    replace = {key: getattr(storage, key).at[step].set(value) for key, value in kwargs.items()}
    return storage.replace(**replace)
