from src.enviroment import Shape
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.trainer import SAETrainer
from src.pod.trajectorystorage import TrajectoryStorage
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface

import gymnasium as gym
import jax


class SimpleWorldModel(WorldModelInterface):
    def __init__(self, env: gym.Env):
        self._model: ModelWrapper = ModelWrapper(StochasticAutoencoder(*Shape()), "autoencoder")
        self._frame_stack = FrameStack(env)
        self._trainer = SAETrainer(self._model)

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        next_frame, reward = self._model.forward(action, self._frame_stack.frames)
        self._frame_stack.add_frame(next_frame)
        return self._frame_stack.frames, reward, False, False, {}

    def reset(self):
        self._frame_stack.reset()
        return self._frame_stack.frames, 0, False, False, {}

    def update(self, data: TrajectoryStorage):
        stack, actions, next_frame = data[:]
        self._trainer.train_step(self._model.params, stack, actions, next_frame)

    def save(self):
        self._model.save("stochastic_autoencoder")

    def load(self):
        self._model.load("stochastic_autoencoder")
