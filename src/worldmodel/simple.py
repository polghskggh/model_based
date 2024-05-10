from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.saetrainer import SAETrainer
from src.pod.hyperparameters import hyperparameters
from src.pod.trajectorystorage import TrajectoryStorage
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface

import gymnasium as gym
import jax


class SimpleWorldModel(WorldModelInterface):
    def __init__(self, env: gym.Env, deterministic: bool = True):
        self._deterministic = deterministic
        self._batch_size = hyperparameters["simple"]["batch_size"]

        if self._deterministic:
            self._model: ModelWrapper = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        else:
            self._model: ModelWrapper = ModelWrapper(StochasticAutoencoder(*Shape()), "autoencoder")

        self._frame_stack = FrameStack(env)
        self._time_step = 0

        if self._deterministic:
            self._trainer = SAETrainer(self._model)

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        next_frame, reward = self._model.forward(action, self._frame_stack.frames)
        self._frame_stack.add_frame(next_frame)
        truncated = self._time_step >= hyperparameters["max_episode_length"] - 1
        return self._frame_stack.frames, reward, False, truncated, {}

    def reset(self):
        self._frame_stack.reset()
        self._time_step = 0
        return self._frame_stack.frames, 0, False, False, {}

    def _deterministic_update(self, stack, actions, next_frame):
        for index in range(0, stack.shape[0], self._batch_size):
            batch = data[index:end_idx]
            teach_pixels = vmap(tile_image)(batch[3])
            teach_reward = batch[2].reshape(-1, 1)
            grads = self._model.train_step((teach_pixels, teach_reward), batch[0], batch[1])
            self._model.apply_grads(grads)

    def update(self, data: TrajectoryStorage):
        for index in range(0, data.size, self._batch_size):
            end_idx = min(index + self._batch_size, stack.shape[0])
            stack, actions, next_frame = data[:]
        if self._deterministic:
            self._deterministic_update(stack, actions, next_frame)
        else:
            self._trainer.train_step(self._model.params, stack, actions, next_frame)

    def save(self):
        self._model.save("stochastic_autoencoder")

    def load(self):
        self._model.load("stochastic_autoencoder")
