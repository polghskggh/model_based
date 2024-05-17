from ctypes import Array

import chex
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
from src.utils.rl import generalized_advantage_estimation


class PPOCritic:
    def __init__(self):
        self._model: ModelWrapper = ModelWrapper(StateValueAtariNN(Shape()[0], 1, True), "ppocritic",
                                                 train_model=StateValueAtariNN(Shape()[0], 1, False))
        self._action_dim: int = Shape()[1]
        self._discount_factor: float = Args().args.discount_factor
        self._lambda = Args().args.gae_lambda

    def calculate_grads(self, states: jax.Array, returns: jax.Array) -> dict:
        """
        Calculate the gradients for the critic model

        :param states: The inputs to the model
        :param returns: The teacher output
        :return:
        """
        grads = self._model.train_step(returns.reshape(-1, 1), states)
        return grads

    def update(self, grads: dict):
        """
        Update the critic model with the given gradients

        :param grads: The gradients to update the model with
        """
        self._model.apply_grads(grads)

    def provide_feedback(self, states: jax.Array, rewards: jax.Array, dones: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Calculate the advantage and returns for the given states, rewards, and dones

        :param states: states in the format (time, batch, height, width, channels)
        :param rewards: rewards in the format (time, batch)
        :param dones: dones in the format (time, batch)

        :return: The lambda advantage and returns
        """
        chex.assert_rank(states, 5)
        chex.assert_rank([rewards, dones], 2)

        values = vmap(self._model.forward)(states)
        values = jnp.squeeze(values)
        discounts = jnp.where(dones, 0, self._discount_factor)
        advantage, returns = generalized_advantage_estimation(values, rewards, discounts, self._lambda)
        return advantage.reshape(-1), returns.reshape(-1)

    @staticmethod
    @jit
    def calculate_advantage(rewards, values, discounts, lambda_):
        values = values.squeeze()
        rewards = rewards.squeeze()
        advantage = truncated_generalized_advantage_estimation(rewards, discounts, lambda_, values)
        return advantage

    def save(self):
        self._model.save()

    def load(self):
        self._model.load("ppocritic")
