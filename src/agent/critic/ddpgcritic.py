from src.agent.actor import ActorInterface
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

    def update_model(self, reward: np.ndarray[float], state: np.ndarray[float], action: np.ndarray[float],
                     next_state: np.ndarray[float], next_action: np.ndarray[float]):

        observed_values: np.ndarray[float] = (
                reward + self._discount_factor * self._model.forward(next_state, next_action).reshape(-1))
        self._model.train_step(observed_values, state, action)
        self.update_target()

    def update_target(self):
        self._target_model.update_polyak(self._polyak, self._model)

    def provide_feedback(self, state: np.ndarray[float], action: np.ndarray[float]) -> np.ndarray[float]:
        return self._model.calculate_gradian_ascent(1, state, action)

    def update_common_head(self, actor: ActorInterface):
        actor.model.params["params"]["cnn"] = self._model.params["params"]["cnn"]
