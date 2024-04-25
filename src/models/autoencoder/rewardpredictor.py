from ctypes import Array

import flax.linen as nn

import jax.numpy as jnp


class RewardPredictor(nn.Module):
    @nn.compact
    def __call__(self, middle: Array, final: Array) -> Array:
        reward_pred = jnp.concat((middle, final), axis=-1)
        reward_pred.reshape(reward_pred.shape[0], -1)
        reward_pred = nn.Dense(128, name="reward_hidden")(reward_pred)
        reward_pred = nn.relu(reward_pred)
        reward_pred = nn.Dense(1, name="reward_final")(reward_pred)
        return reward_pred
