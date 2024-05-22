import jax
from jax import vmap

from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import TransitionStorage
from src.singletons.hyperparameters import Args
from src.trainer.saetrainer import SAETrainer
from src.utils.rl import tile_image, reverse_tile_image
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface


class SimpleWorldModel(WorldModelInterface):
    def __init__(self, deterministic: bool = False):
        self._deterministic = deterministic
        self._batch_size = Args().args.batch_size
        self._action_space = Shape()[1]
        self._rollout_length = Args().args.sim_trajectory_length

        if self._deterministic:
            self._model: ModelWrapper = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        else:
            self._model: ModelWrapper = ModelWrapper(StochasticAutoencoder(*Shape()), "autoencoder")
            self._trainer = SAETrainer(self._model)

        self._frame_stack = None
        self._time_step = 0

    def step(self, actions: jax.Array) -> (jax.Array, float, bool, bool, dict):
        next_frames, rewards = self._model.forward(self._frame_stack.frames, actions)
        next_frames = vmap(vmap(reverse_tile_image))(next_frames)
        self._frame_stack.add_frame(next_frames)
        self._time_step += 1
        truncated = self._time_step >= self._rollout_length
        return self._frame_stack.frames, rewards, False, truncated, {}

    def reset(self):
        self._frame_stack.reset()
        self._time_step = 0
        return self._frame_stack.frames, {}

    def _deterministic_update(self, stack, actions, rewards, next_frame):
        next_frame = vmap(tile_image)(next_frame)
        batch_size = Args().args.batch_size
        for start_idx in range(0, stack.shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            grads = self._model.train_step((next_frame[batch_slice], rewards[batch_slice]), stack[batch_slice], actions[batch_slice])
            self._model.apply_grads(grads)

    def _stochastic_update(self, stack, actions, rewards, next_frame):
        self._model.params = self._trainer.train_step(self._model.params, stack, actions, rewards, next_frame)

    def update(self, storage: TransitionStorage):
        update_fn = self._deterministic_update if self._deterministic else self._stochastic_update
        for _ in range(Args().args.num_epochs):
            update_fn(storage.observations, storage.actions, storage.rewards, storage.next_observations)

        self._frame_stack = FrameStack(storage)

    def save(self):
        self._model.save()

    def load(self):
        self._model.load("stochastic_autoencoder")

    def wrap_env(self, env):
        return env
