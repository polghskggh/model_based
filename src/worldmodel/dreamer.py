import operator
import time
from functools import reduce

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr

from src.enviroment import Shape
from src.models.autoencoder.encoder import Encoder
from src.models.dreamer.observation import ObservationModel
from src.models.dreamer.representation import RepresentationModel
from src.models.dreamer.reward import PredictModel
from src.models.dreamer.transition import TransitionModel
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import store, DreamerStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log
from src.trainer.dreamertrainer import DreamerTrainer
from src.utils.rl import process_output
from src.worldmodel.worldmodelinterface import WorldModelInterface


class Dreamer(WorldModelInterface):
    def __init__(self):
        Args().args.bottleneck_dims = (2, 2, 64)
        self.num_agents = Args().args.num_agents
        self.belief_size = Args().args.belief_size
        self.state_size = Args().args.state_size
        self.embedding_size = reduce(operator.mul, Args().args.bottleneck_dims)
        self.hidden_size = Args().args.hidden_size
        self.observation_size = Shape()[0]
        self.action_size = Shape()[1]

        self.initial_beliefs = jnp.zeros(self.belief_size)
        self.initial_states = jnp.zeros(self.state_size)

        self.prev_belief = jnp.zeros((self.num_agents, self.belief_size))
        self.prev_state = jnp.zeros((self.num_agents, self.state_size))

        # Initialise model parameters randomly
        representation_model = ModelWrapper(RepresentationModel(self.belief_size, self.state_size,
                                                                self.action_size, self.hidden_size,
                                                                self.embedding_size, Shape()[0]),
                                            "representation")
        transition_model = ModelWrapper(TransitionModel(self.belief_size, self.state_size, self.hidden_size),
                                        "transition")
        observation_model = ModelWrapper(ObservationModel(self.belief_size, self.state_size,
                                                          self.embedding_size, self.observation_size),
                                         "observation")

        encoder_model = ModelWrapper(Encoder(Args().args.bottleneck_dims[2]), "encoder")

        reward_model = ModelWrapper(PredictModel(self.hidden_size, Args().args.rewards), "reward")

        self.models = {"representation": representation_model,
                       "observation": observation_model,
                       "reward": reward_model,
                       "transition": transition_model,
                       "encoder": encoder_model}

        if Args().args.predict_dones:
            dones_model = ModelWrapper(PredictModel(self.hidden_size, 2), "reward")
            self.models["dones"] = dones_model

        self.trainer = DreamerTrainer(self.models)

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        start_time = time.time()
        self.prev_belief, self.prev_state, _, _ = self.models["transition"].forward(self.prev_state, action,
                                                                                    self.prev_belief)
        imagined_reward_logits = self.models["reward"].forward(self.prev_belief, self.prev_state)
        imagined_reward = process_output(imagined_reward_logits)

        if Args().args.predict_dones:
            dones = self.models["dones"].forward(self.prev_belief, self.prev_state)
            dones = process_output(dones)
        else:
            dones = jnp.zeros(imagined_reward.shape, dtype=bool)

        log({"Step time": (time.time() - start_time) / action.shape[0]})
        return (jnp.append(self.prev_belief, self.prev_state, -1), imagined_reward, dones,
                jnp.zeros(imagined_reward.shape, dtype=bool), {})

    def reset(self) -> (jax.Array, float, bool, bool, dict):
        key = Key().key(1)
        idx = jr.choice(key, self.initial_beliefs.shape[0], (self.num_agents,), False)
        self.prev_state = self.initial_states[idx]
        self.prev_belief = self.initial_beliefs[idx]
        return jnp.append(self.prev_belief, self.prev_state, axis=-1), {}

    def update(self, data):
        observations, actions, rewards, dones = data.observations, data.actions, data.rewards, data.dones

        self.models = self.trainer.train_step(data.beliefs[0], data.states[0], observations, actions, rewards, dones)

        self.initial_beliefs = data.beliefs.reshape(-1, self.belief_size)
        self.initial_states = data.states.reshape(-1, self.state_size)

    def wrap_env(self, envs):
        return DreamerWrapper(envs, self.models["representation"], self.models["encoder"])


class DreamerWrapper(gym.Wrapper):
    representation_model: ModelWrapper
    prev_state: jax.Array
    prev_belief: jax.Array

    def __init__(self, env, representation_model: ModelWrapper, encoder_model: ModelWrapper):
        super().__init__(env)
        self.representation_model = representation_model
        self.encoder_model = encoder_model
        batch_shape = (Args().args.trajectory_length,
                       Args().args.num_envs)
        self.prev_belief = jnp.zeros((Args().args.num_envs, Args().args.belief_size))
        self.prev_state = jnp.zeros((Args().args.num_envs, Args().args.state_size))
        self.timestep = 0
        self.storage = DreamerStorage(observations=jnp.zeros(batch_shape + Shape()[0]),
                                      actions=jnp.zeros(batch_shape),
                                      rewards=jnp.zeros(batch_shape),
                                      dones=jnp.zeros(batch_shape),
                                      beliefs=jnp.zeros(batch_shape + (Args().args.belief_size,)),
                                      states=jnp.zeros(batch_shape + (Args().args.state_size,)))

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return jnp.append(self.prev_belief, self.prev_state, axis=-1), info

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)

        encoded_observation, _ = self.encoder_model.forward(observation)
        belief, state, _, _, _, _ = self.representation_model.forward(self.prev_state, action,
                                                                      self.prev_belief, encoded_observation)
        self.storage = store(self.storage, self.timestep, observations=observation, actions=action, rewards=reward,
                             dones=term | trunc, beliefs=self.prev_belief, states=self.prev_state)

        self.prev_belief = belief
        self.prev_state = state

        self.timestep += 1
        self.timestep %= Args().args.trajectory_length
        return jnp.append(belief, state, axis=-1), reward, term, trunc, info

