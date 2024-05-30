import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class PPOStorage:
    observations: jnp.array
    rewards: jnp.array
    actions: jnp.array
    log_probs: jnp.array
    dones: jnp.array
    values: jnp.ndarray


@dataclass
class TransitionStorage:
    observations: jnp.array
    actions: jnp.array
    rewards: jnp.array
    next_observations: jnp.array


@dataclass
class TrajectoryStorage:
    observations: jnp.array
    actions: jnp.array
    rewards: jnp.array


@dataclass
class DreamerStorage:
    observations: jnp.array
    actions: jnp.array
    rewards: jnp.array
    beliefs: jnp.array
    states: jnp.array


def store(storage, step: slice | int, **kwargs):
    replace = {key: getattr(storage, key).at[step].set(value) for key, value in kwargs.items()}
    return storage.replace(**replace)
