from ctypes import Array

import jax.numpy as jnp
from flax import linen as nn

from src.agent.critic import CriticInterface
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters

from src.trainer.critictrainer import DDPGCriticTrainer


class DQNNetwork(CriticInterface):

    def __init__(self, model: nn.Module):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "dqncritic")
        self._target_model: ModelWrapper = ModelWrapper(model, "dqncritic")

        self._discount_factor: float = hyperparameters["dqn"]["discount_factor"]
        self._target_update_period = hyperparameters["dqn"]["target_update_period"]
        self._iteration = 0

    def calculate_grads(self, state: Array[float], action: Array[float], reward: float,
                        next_state: Array[float], next_action: Array[float]) -> dict:
        observed_values: Array[float] = (
                reward + self._discount_factor * self._target_model.forward(next_state, next_action).reshape(-1))
        observed_values = jnp.expand_dims(observed_values, 1)

        return self._model.train_step(observed_values, state, action)

    def update(self, grads: dict):
        self._model.apply_grads(grads)

        self._iteration += 1
        if self._iteration % self._target_update_period == 0:
            self._target_model.params = self._model.params

    def provide_feedback(self, state: Array, action: Array) -> Array:
        pass

    def save(self):
        self._model.save("critic")
        self._target_model.save("target_critic")

    def load(self):
        self._model.load("critic")
        self._target_model.load("target_critic")
