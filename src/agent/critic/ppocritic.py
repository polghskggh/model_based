from ctypes import Array

import jax
import rlax
from flax import linen as nn
from jax import vmap, lax
from rlax import truncated_generalized_advantage_estimation

from src.agent.critic import CriticInterface
from src.enviroment import Shape
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters
import jax.numpy as jnp

from src.trainer.critictrainer import PPOCriticTrainer


class PPOCritic(CriticInterface):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "ppocritic")
        self._action_dim: int = Shape()[1]
        self._trainer = PPOCriticTrainer(self._model.model)
        self._bootstrapped_values = None
        self._discount_factor: float = hyperparameters["ppo"]["discount_factor"]
        self._lambda = hyperparameters["ppo"]["lambda"]

    def __update_bootstrap_values(self, states: jax.Array):
        if self._bootstrapped_values is None:
            self._bootstrapped_values = vmap(self._model.forward)(states)

    def calculate_grads(self, states: jax.Array, reward: jax.Array) -> dict:
        rewards_to_go = self.__batch_calculate_rewards_to_go(reward, states)
        rewards_to_go = jnp.expand_dims(rewards_to_go, axis=-1)
        states = lax.slice_in_dim(states, 0, -1, axis=1)
        grads = self.__batch_train_step(rewards_to_go, states)
        return grads

    def __batch_train_step(self, rewards: jax.Array, states: jax.Array) -> dict:
        rewards = rewards.reshape(-1, *rewards.shape[2:])
        states = states.reshape(-1, *states.shape[2:])

        grads = self._model.train_step(rewards, states)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)
        self._bootstrapped_values = None

    def provide_feedback(self, states: Array, rewards: Array) -> Array:
        return self.__batch_calculate_advantage(states, rewards)

    def __batch_calculate_advantage(self, states: jax.Array, rewards: jax.Array) -> jax.Array:
        self.__update_bootstrap_values(states)
        advantage_fun = vmap(PPOCriticTrainer.calculate_advantage, in_axes=(0, 0, None, None))
        return advantage_fun(rewards, self._bootstrapped_values, self._discount_factor, self._lambda)

    def __batch_calculate_rewards_to_go(self, rewards: jax.Array, states: jax.Array) -> jax.Array:
        self.__update_bootstrap_values(states)
        return vmap(self.__calculate_rewards_to_go, (0, 0))(rewards, self._bootstrapped_values)

    @staticmethod
    def __calculate_advantage(rewards, values, discount_factor, lambda_par):
        values = values.reshape(-1)
        rewards = rewards.reshape(-1)

        discounts = discount_factor * jnp.ones_like(rewards)
        advantage = truncated_generalized_advantage_estimation(rewards, discounts, lambda_par, values)
        return advantage

    def __calculate_rewards_to_go(self, rewards: jax.Array, values: jax.Array) -> float:
        values = values[1:].reshape(-1)  # remove the first value
        rewards = rewards.reshape(-1)

        discount = self._discount_factor * jnp.ones_like(rewards)
        return rlax.discounted_returns(rewards, discount, values)
