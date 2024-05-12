import jax
from jax import vmap
from rlax import one_hot

from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.saetrainer import SAETrainer
from src.pod.hyperparameters import hyperparameters
from src.pod.trajectorystorage import TrajectoryStorage
from src.utils.tiling import tile_image, reverse_tile_image
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface


class SimpleWorldModel(WorldModelInterface):
    def __init__(self, deterministic: bool = False):
        self._deterministic = deterministic
        self._batch_size = hyperparameters["simple"]["batch_size"]
        self._action_space = Shape()[1]
        self._parallel_agents = hyperparameters["simple"]["parallel_agents"]

        if self._deterministic:
            self._model: ModelWrapper = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        else:
            self._model: ModelWrapper = ModelWrapper(StochasticAutoencoder(*Shape()), "autoencoder")

        self._frame_stack = None
        self._time_step = 0

        if not self._deterministic:
            self._trainer = SAETrainer(self._model)

    def step(self, actions: jax.Array) -> (jax.Array, float, bool, bool, dict):
        next_frames, rewards = self._model.forward(self._frame_stack.frames, actions)
        next_frames = vmap(vmap(reverse_tile_image))(next_frames)
        self._frame_stack.add_frames(next_frames)
        self._time_step += 1
        truncated = self._time_step >= hyperparameters["simple"]["rollout_length"]
        return self._frame_stack.frames, rewards, False, truncated, {}

    def reset(self):
        self._frame_stack.reset()
        self._time_step = 0
        return self._frame_stack.frames, {}

    def _deterministic_update(self, stack, actions, rewards, next_frame):
        teach_pixels = vmap(tile_image)(next_frame)
        grads = self._model.train_step((teach_pixels, rewards), stack, actions)
        self._model.apply_grads(grads)

    def _stochastic_update(self, stack, actions, rewards, next_frame):
        self._model.params = self._trainer.train_step(self._model.params, stack, actions, rewards, next_frame)

    def update(self, data: TrajectoryStorage):
        for stack, actions, rewards, next_frame in data.batched_data():
            rewards = rewards.reshape(-1, 1)
            if self._deterministic:
                self._deterministic_update(stack, actions, rewards, next_frame)
            else:
                self._stochastic_update(stack, actions, rewards, next_frame)

        self._frame_stack = FrameStack(data)

    def save(self):
        self._model.save("stochastic_autoencoder")

    def load(self):
        self._model.load("stochastic_autoencoder")
