import time

import gym
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import TransitionStorage, store
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log
from src.trainer.saetrainer import SAETrainer
from src.utils.rl import process_output
from src.worldmodel.framestack import FrameStack
from src.worldmodel.worldmodelinterface import WorldModelInterface


class SimpleWorldModel(WorldModelInterface):
    def __init__(self, deterministic: bool = False):
        self._deterministic = deterministic
        self._batch_size = Args().args.batch_size
        self._action_space = Shape()[1]
        self._rollout_length = Args().args.sim_trajectory_length

        if self._deterministic:
            self._model: ModelWrapper = ModelWrapper(AutoEncoder(*Shape(), deterministic=True), "autoencoder",
                                                     AutoEncoder(*Shape(), deterministic=False))
        else:
            self._model: ModelWrapper = ModelWrapper(StochasticAutoencoder(*Shape()), "autoencoder")
            self._trainer = SAETrainer(self._model)

        self._frame_stack = None
        self.predict_dones = Args().args.predict_dones

    def step(self, actions: jax.Array) -> (jax.Array, float, bool, bool, dict):
        start_time = time.time()
        prediction = self._model.forward(self._frame_stack.frames, actions)
        dones = None
        if self.predict_dones:
            next_frames, rewards_logits, dones = prediction
        else:
            next_frames, rewards_logits = prediction

        if Args().args.categorical_image:
            next_frames = jnp.argmax(next_frames, axis=-1, keepdims=True)

        rewards = process_output(rewards_logits)
        rewards += Args().args.min_reward
        dones = process_output(dones) if dones is not None else jnp.zeros_like(rewards, dtype=bool)

        self._frame_stack.add_frame(next_frames)

        log({"Step time": (time.time() - start_time) / actions.shape[0]})
        return self._frame_stack.frames, rewards, dones, jnp.zeros_like(rewards, dtype=bool), {}

    def reset(self):
        self._frame_stack.reset()
        return self._frame_stack.frames, {}

    def _deterministic_update(self, stack, actions, rewards, dones, next_frame):
        batch_size = Args().args.batch_size
        epoch_size = stack.shape[0]

        epoch_indices = jr.permutation(Key().key(), epoch_size)
        for start_idx in range(0, epoch_size, batch_size):
            end_idx = start_idx + batch_size
            batch_slice = epoch_indices[start_idx:end_idx]
            target = (next_frame[batch_slice], rewards[batch_slice])
            if Args().args.predict_dones:
                target += (dones[batch_slice], )

            grads = self._model.train_step(target, stack[batch_slice], actions[batch_slice])
            self._model.apply_grads(grads)

    def _stochastic_update(self, stack, actions, rewards, next_frame):
        self._model.params = self._trainer.train_step(self._model.params, stack, actions, rewards, next_frame)

    def update(self, storage: TransitionStorage):
        self._frame_stack = FrameStack(storage.observations)
        update_fn = self._deterministic_update if self._deterministic else self._stochastic_update
        print("updating with reward", jnp.mean(storage.rewards))
        print("updating with dones", jnp.mean(storage.dones))
        for _ in range(Args().args.num_epochs):
            update_fn(storage.observations, storage.actions, storage.rewards, storage.dones, storage.next_observations)

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
                                          dones=jnp.zeros(batch_shape),
                                          next_observations=jnp.zeros(batch_shape + (Shape()[0][0], Shape()[0][1],
                                                                       self.n_channels)))

    def reset(self, **kwargs):
        self.last_observation, info = self.env.reset(**kwargs)
        return self.last_observation, info

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)
        next_observations = lax.slice_in_dim(observation, (Args().args.frame_stack - 1) * self.n_channels,
                                             None, axis=-1)

        store_slice = slice(self._timestep * Args().args.num_envs, (self._timestep + 1) * Args().args.num_envs)
        self._storage = store(self._storage, store_slice, observations=self.last_observation, actions=action,
                              rewards=reward, dones=term | trunc, next_observations=next_observations)
        self._timestep += 1
        self._timestep %= Args().args.trajectory_length

        self.last_observation = observation
        return observation, reward, term, trunc, info

    @property
    def storage(self):
        return self._storage
