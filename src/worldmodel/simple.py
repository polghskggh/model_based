from typing import Tuple

import gym
import jax
from jax import vmap, lax
import jax.numpy as jnp

from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import TransitionStorage, store
from src.singletons.hyperparameters import Args
from src.trainer.saetrainer import SAETrainer
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface
import flax.linen as nn


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
        next_frames, rewards_logits = self._model.forward(self._frame_stack.frames, actions)
        next_frames = jnp.argmax(nn.softmax(next_frames), axis=-1, keepdims=True)
        rewards = jnp.argmax(nn.softmax(rewards_logits), axis=-1)
        self._frame_stack.add_frame(next_frames)
        self._time_step += 1
        rewards = rewards.squeeze()
        return (self._frame_stack.frames, rewards, jnp.zeros(rewards.shape, dtype=bool),
                jnp.zeros(rewards.shape, dtype=bool), {})

    def reset(self):
        self._frame_stack.reset()
        self._time_step = 0
        return self._frame_stack.frames, {}

    def _deterministic_update(self, stack, actions, rewards, next_frame):
        batch_size = Args().args.batch_size
        for start_idx in range(0, stack.shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            grads = self._model.train_step((next_frame[batch_slice], rewards[batch_slice]),
                                           stack[batch_slice], actions[batch_slice])
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
        return SimpleWrapper(env)


class SimpleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        batch_shape = (Args().args.trajectory_length * Args().args.num_envs, )
        self.last_observation = None
        self._timestep = 0
        self.n_channels = Shape()[0][2] // Args().args.frame_stack
        self._storage = TransitionStorage(observations=jnp.zeros(batch_shape + Shape()[0]),
                                          actions=jnp.zeros(batch_shape),
                                          rewards=jnp.zeros(batch_shape),
                                          next_observations=jnp.zeros(batch_shape + (Shape()[0][0], Shape()[0][1],
                                                                       self.n_channels)))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_observation = observation
        self._timestep = 0
        return observation, info

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)

        next_observations = lax.slice_in_dim(observation, (Args().args.frame_stack - 1) * self.n_channels,
                                             None, axis=-1)

        store_slice = slice(self._timestep * Args().args.num_envs, (self._timestep + 1) * Args().args.num_envs)
        self._storage = store(self._storage, store_slice, observations=self.last_observation, actions=action,
                              rewards=reward, next_observations=next_observations)
        self.last_observation = observation
        return observation, reward, term, trunc, info

    @property
    def storage(self):
        return self._storage
