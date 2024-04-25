from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters
from src.pod.replaybuffer import ReplayBuffer
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface
import gymnasium as gym
import jax


class DeterministicSimple(WorldModelInterface):
    def __init__(self, env: gym.Env):
        self._model: ModelWrapper = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        self._frame_stack = FrameStack(env)
        self._rollout_length = hyperparameters["world"]["rollout_length"]

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        next_frame, reward = self._model.forward(action, self._frame_stack.frames)
        self._frame_stack.add_frame(next_frame)
        return self._frame_stack.frames, reward, False, False, {}

    def reset(self):
        self._frame_stack.reset()
        return self._frame_stack.frames

    def update(self, data: ReplayBuffer):
        data.data()

        self._model.train(data, self._rollout_length)

    def save(self):
        self._model.save("autoencoder")

    def load(self):
        self._model.load("autoencoder")
