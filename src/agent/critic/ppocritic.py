from ctypes import Array

import jax
import rlax
from flax import linen as nn
from jax import vmap, jit, lax

from src.agent.critic import CriticInterface
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.critictrainer import PPOCriticTrainer
from src.pod.hyperparameters import hyperparameters
import jax.numpy as jnp


class PPOCritic(CriticInterface):
    def __init__(self, model: nn.Module, polyak: float = 0.995, action_dim: int = 4):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "ppocritic")

        self._discount_factor: float = hyperparameters["ppo"]["discount_factor"]
        self._polyak: float = polyak
        self._action_dim: int = action_dim
        self._trainer = PPOCriticTrainer(self._model.model)
        self._bootstrapped_values = None

    def __update_bootstrap_values(self, states: jax.Array) -> None:
        print("states", states.shape)
        if self._bootstrapped_values is None:
            self._bootstrapped_values = vmap(self._model.forward)(states)

    def __calculate_advantage(self, states: jax.Array, rewards: jax.Array) -> jax.Array:
        self.__update_bootstrap_values(states)
        advantage = self._trainer.train_step(rewards, self._bootstrapped_values)
        return advantage

    def __calculate_rewards_to_go(self, rewards: jax.Array, values: jax.Array) -> float:
        values = values[1:].reshape(-1)  # remove the first value
        rewards = rewards.reshape(-1)

        discount = self._discount_factor * jnp.ones_like(rewards)
        return rlax.discounted_returns(rewards, discount, values)

    def __batch_calculate_rewards_to_go(self, rewards: jax.Array, states: jax.Array) -> jax.Array:
        self.__update_bootstrap_values(states)
        return vmap(self.__calculate_rewards_to_go, (0, 0))(rewards, self._bootstrapped_values)

    def calculate_grads(self, states: jax.Array, reward: jax.Array) -> dict:
        rewards_to_go = self.__batch_calculate_rewards_to_go(reward, states)
        rewards_to_go = jnp.expand_dims(rewards_to_go, axis=-1)
        states = lax.slice_in_dim(states, 0, -1, axis=1)
        grads = vmap(self.__batch_train_step, in_axes=(0, 0))(rewards_to_go, states)

        return grads

    def __batch_train_step(self, rewards: jax.Array, states: jax.Array) -> jax.Array:
        grads = self._model.train_step(rewards, states)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)
        self._bootstrapped_values = None

    def provide_feedback(self, states: Array, rewards: Array) -> Array:
        return self.__calculate_advantage(states, rewards)
