from ctypes import Array

import jax
import jax.numpy as jnp
import rlax
from flax import linen as nn
from jax import vmap, jit
from rlax import truncated_generalized_advantage_estimation

from src.agent.critic import CriticInterface
from src.enviroment import Shape
from src.models.actorcritic.atarinn import StateValueAtariNN
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters


class PPOCritic(CriticInterface):
    def __init__(self):
        self._model: ModelWrapper = ModelWrapper(StateValueAtariNN(Shape()[0], 1, True), "ppocritic",
                                                 train_model=StateValueAtariNN(Shape()[0], 1, False))
        self._action_dim: int = Shape()[1]
        self._bootstrapped_values = None
        self._discount_factor: float = hyperparameters["ppo"]["discount_factor"]
        self._lambda = hyperparameters["ppo"]["lambda"]

    def __update_bootstrap_values(self, states: jax.Array):
        if self._bootstrapped_values is None:
            self._bootstrapped_values = vmap(self._model.forward)(states)

    def calculate_grads(self, states: jax.Array, returns: jax.Array) -> dict:
        grads = self._model.train_step(returns.reshape(-1, 1), states)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)
        self._bootstrapped_values = None

    def provide_feedback(self, states: Array, rewards: Array, dones: jax.Array) -> Array:
        self.__update_bootstrap_values(states)
        advantage_fun = jit(vmap(PPOCritic.calculate_advantage, in_axes=(0, 0, 0, None)))
        discounts = jnp.where(dones, 0, self._discount_factor)
        return advantage_fun(rewards, self._bootstrapped_values, discounts, self._lambda)

    @staticmethod
    def calculate_advantage(rewards, values, discounts, lambda_):
        values = values.reshape(-1)
        rewards = rewards.reshape(-1)

        advantage = truncated_generalized_advantage_estimation(rewards, discounts, lambda_, values)
        return advantage

    def calculate_rewards_to_go(self, rewards: jax.Array, states: jax.Array, dones: jax.Array) -> jax.Array:
        self.__update_bootstrap_values(states)
        discounts = jnp.where(dones, 0, self._discount_factor)
        return jit(vmap(PPOCritic.__calculate_rewards_to_go, (0, 0, 0)))(rewards, self._bootstrapped_values,
                                                                         discounts)

    @staticmethod
    def __calculate_rewards_to_go(rewards: jax.Array, values: jax.Array, discounts: jax.Array) -> float:
        values = values[1:].reshape(-1)          # skip the first value
        rewards = rewards.reshape(-1)
        return rlax.discounted_returns(rewards, discounts, values)



