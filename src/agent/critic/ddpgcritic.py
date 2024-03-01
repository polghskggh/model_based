from src.agent.actor import ActorInterface
from src.agent.critic import CriticInterface
from src.models import ModelWrapper
from flax import linen as nn
import numpy as np

from src.resultwriter import ModelWriter


class DDPGCritic(CriticInterface):
    def __init__(self, model: nn.Module, discount_factor: float, polyak: float = 0.995, action_dim: int = 4):
        super().__init__()
        writer = ModelWriter("critic", "critic_loss")
        self._model: ModelWrapper = ModelWrapper(model, writer)
        self._target_model: ModelWrapper = ModelWrapper(model, writer)

        self._discount_factor: float = discount_factor
        self._polyak: float = polyak
        self._action_dim: int = action_dim

    def update_model(self, reward: np.ndarray[float], state: np.ndarray[float], action: np.ndarray[float],
                     next_state: np.ndarray[float], next_action: np.ndarray[float]):

        state_action: np.ndarray[float] = np.append(state, action, axis=1)
        new_state_action: np.ndarray[float] = np.append(next_state, next_action, axis=1)

        observed_values: np.ndarray[float] = (
                reward + self._discount_factor * self._target_model.forward(new_state_action).reshape(-1))

        self._model.train_step(state_action, observed_values)
        self.update_target()

    def update_target(self):
        self._target_model.update_polyak(self._polyak, self._model)

    def provide_feedback(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        return self._model.calculate_gradian_ascent(actions)[:self._action_dim]
