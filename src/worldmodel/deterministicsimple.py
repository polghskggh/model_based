from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters
from src.pod.replaybuffer import ReplayBuffer
from src.pod.trajectorystorage import TrajectoryStorage
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface
import gymnasium as gym
import jax


class DeterministicSimple(WorldModelInterface):
    def __init__(self, env: gym.Env):
        self._model: ModelWrapper = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        self._frame_stack = FrameStack(env)
        self._batch_size = hyperparameters["world"]["batch_size"]

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        next_frame, reward = self._model.forward(action, self._frame_stack.frames)
        self._frame_stack.add_frame(next_frame)
        return self._frame_stack.frames, reward, False, False, {}

    def reset(self):
        self._frame_stack.reset()
        return self._frame_stack.frames, 0, False, False, {}

    def update(self, data: TrajectoryStorage):
        for index in range(0, data.size, self._batch_size):
            end_idx = min(index + self._batch_size, data.size)
            batch = data[index:end_idx]
            grads = self._model.train_step((batch[3], batch[2]), batch[0], batch[1])
            self._model.apply_grads(grads)

    def save(self):
        self._model.save("autoencoder")

    def load(self):
        self._model.load("autoencoder")
