from ctypes import Array

from flax import linen as nn

from src.agent.critic import CriticInterface
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.critictrainer import DDPGCriticTrainer
from src.pod.hyperparameters import hyperparameters
import jax.numpy as jnp


class DDPGCritic(CriticInterface):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "critic")
        self._target_model: ModelWrapper = ModelWrapper(model, "critic")

        self._discount_factor: float = hyperparameters["ddpg"]["discount_factor"]
        self._polyak: float = hyperparameters["ddpg"]["polyak"]
        self._trainer = DDPGCriticTrainer(self._model.model)

    def calculate_grads(self, state: Array[float], action: Array[float], reward: float,
                        next_state: Array[float], next_action: Array[float]) -> dict:
        observed_values: Array[float] = (
                reward + self._discount_factor * self._target_model.forward(next_state, next_action).reshape(-1))
        observed_values = jnp.expand_dims(observed_values, 1)

        return self._model.train_step(observed_values, state, action)

    def update(self, grads: dict):
        self._model.apply_grads(grads)
        self._target_model.update_polyak(self._polyak, self._model)

    def provide_feedback(self, state: Array, action: Array) -> Array:
        return self._trainer.train_step(self._model.params, state, action)
