from ctypes import Array

import rlax
from jax import value_and_grad

from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer


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
