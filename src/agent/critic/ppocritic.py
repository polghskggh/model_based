from ctypes import Array

import jax
from flax import linen as nn

from src.agent.critic import CriticInterface
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.critictrainer import PPOCriticTrainer
from src.pod.hyperparameters import hyperparameters


class PPOCritic(CriticInterface):
    def __init__(self, model: nn.Module, polyak: float = 0.995, action_dim: int = 4):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "critic")

        self._discount_factor: float = hyperparameters["ppo"]["discount_factor"]
        self._polyak: float = polyak
        self._action_dim: int = action_dim
        self._trainer = PPOCriticTrainer(self._model.model)
        self._advantage = None

    def _calculate_advantage(self, states: Array, rewards: Array) -> Array:
        advantage = self._trainer.train_step(rewards, self._model.forward(states))
        return advantage

    def calculate_grads(self, states: jax.Array, reward: jax.Array) -> jax.Array:
        advantage: Array[float] = self._calculate_advantage(states, reward)
        grads = self._model.train_step(advantage, states)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    def provide_feedback(self, states: Array, rewards: Array) -> Array:
        return self._calculate_advantage(states, rewards)
