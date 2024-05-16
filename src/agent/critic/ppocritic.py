from ctypes import Array

import jax
import jax.numpy as jnp
import rlax
from flax import linen as nn
from jax import vmap, jit
from rlax import truncated_generalized_advantage_estimation

from src.enviroment import Shape
from src.models.actorcritic.atarinn import StateValueAtariNN
from src.models.modelwrapper import ModelWrapper
from src.singletons.hyperparameters import Args


class PPOCritic:
    def __init__(self):
        self._model: ModelWrapper = ModelWrapper(StateValueAtariNN(Shape()[0], 1, True), "ppocritic",
                                                 train_model=StateValueAtariNN(Shape()[0], 1, False))
        self._action_dim: int = Shape()[1]
        self._discount_factor: float = Args().args.discount_factor
        self._lambda = Args().args.gae_lambda

    def calculate_grads(self, states: jax.Array, returns: jax.Array) -> dict:
        grads = self._model.train_step(returns.reshape(-1, 1), states)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    def provide_feedback(self, states: Array, rewards: Array, dones: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = vmap(self._model.forward)(states)
        advantage_fun = vmap(PPOCritic.calculate_advantage, in_axes=(0, 0, 0, None))
        discounts = jnp.where(dones, 0, self._discount_factor)
        advantage = advantage_fun(rewards, values, discounts, self._lambda)
        return advantage, advantage + values

    @staticmethod
    @jit
    def calculate_advantage(rewards, values, discounts, lambda_):
        values = values.squeeze()
        rewards = rewards.squeeze()
        advantage = truncated_generalized_advantage_estimation(rewards, discounts, lambda_, values)
        return advantage

    def save(self):
        pass

    def load(self):
        pass
