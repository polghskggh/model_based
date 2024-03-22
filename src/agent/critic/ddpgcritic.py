from src.agent.actor.actorinterface import ActorInterface
from src.agent.critic import CriticInterface
from src.models.modelwrapper import ModelWrapper
from flax import linen as nn
import numpy as np


class DDPGCritic(CriticInterface):
    def __init__(self, model: nn.Module, discount_factor: float, polyak: float = 0.995, action_dim: int = 4):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "critic")
        self._target_model: ModelWrapper = ModelWrapper(model, "critic")

        self._discount_factor: float = discount_factor
        self._polyak: float = polyak
        self._action_dim: int = action_dim
        print(self._model)

    def calculate_grads(self, reward: np.ndarray[float], state: np.ndarray[float], action: np.ndarray[float],
                        next_state: np.ndarray[float], next_action: np.ndarray[float]) -> np.ndarray[float]:
        observed_values: np.ndarray[float] = (
                reward + self._discount_factor * self._target_model.forward(next_state, next_action).reshape(-1))
        return self._model.train_step(observed_values, state, action)

    def update(self, grads: np.ndarray[float]):
        self._model.apply_grads(grads)
        self._target_model.update_polyak(self._polyak, self._model)

    def provide_feedback(self, actor: ActorInterface, states: np.ndarray[float]) -> dict:
        return self._model.actor_grads(actor.model, states)
