import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class Storage:
    observations: jnp.array
    rewards: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array


def store(storage: Storage, step: slice | int, **kwargs):
    replace = {key: getattr(storage, key).at[step].set(value) for key, value in kwargs.items()}
    return storage.replace(**replace)
