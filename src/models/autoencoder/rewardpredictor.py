from ctypes import Array

import flax.linen as nn
import jax.lax

import jax.numpy as jnp

from src.singletons.hyperparameters import Args


class RewardPredictor(nn.Module):
    @nn.compact
    def __call__(self, middle: Array, final: Array) -> Array:
        middle = middle.reshape(middle.shape[0], -1)
        final = jax.lax.reduce(final, 0.0, lambda x, y: x + y, (1, 2))
        reward_pred = jnp.concat((middle, final), axis=-1)
        reward_pred = nn.Dense(128, name="reward_hidden")(reward_pred)
        reward_pred = nn.relu(reward_pred)
        reward_pred = nn.Dense(Args().args.rewards, name="reward_final")(reward_pred)
        return reward_pred
