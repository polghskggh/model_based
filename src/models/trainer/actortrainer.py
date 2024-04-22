from ctypes import Array

import jax
import rlax
from jax import value_and_grad, lax

from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer
from src.pod.hyperparameters import hyperparameters


class DDPGActorTrainer(Trainer):
    def __init__(self, model: ModelWrapper):
        super().__init__()
        self._model = model

    def train_step(self, params, states: Array[float], action_grads: Array[float]):
        grad_fun = value_and_grad(DDPGActorTrainer.ddpg_grad, 1)
        loss, grads = grad_fun(self._model, params, states, action_grads)
        return grads

    @staticmethod
    def ddpg_grad(model, params, states, action_grads):
        actions = model.apply(params, states)
        loss = rlax.dpg_loss(actions, action_grads)
        return loss


class PPOActorTrainer(Trainer):
    def __init__(self, model: ModelWrapper):
        super().__init__()
        self._model = model
        self._clip_threshold = hyperparameters["ppo"]["clip_threshold"]

    def train_step(self, params, states: Array[float], advantage: Array[float]):
        grad_fun = value_and_grad(PPOActorTrainer.ppo_grad, 1)
        loss, grads = grad_fun(self._model, params, states, advantage, self._clip_threshold)
        grads = jax.tree_util.tree_map(lambda x: -x, grads)
        return grads

    @staticmethod
    def ppo_grad(model, params: dict, states: jax.Array, advantage: float, action_index: int, epsilon: float):
        policy = model.apply(params, states)[action_index]
        policy_old = lax.stop_gradient(policy)

        if advantage > 0:
            clipped_loss = min(policy / policy_old, 1 + epsilon) * advantage
        else:
            clipped_loss = max(policy / policy_old, 1 - epsilon) * advantage

        return clipped_loss
