from typing import Tuple

import jax
import jax.numpy as jnp

from src.enviroment import Shape
from src.models.dreamer.observation import ObservationModel
from src.models.dreamer.representation import RepresentationModel
from src.models.dreamer.reward import RewardModel
from src.models.dreamer.transition import TransitionModel
from src.models.modelwrapper import ModelWrapper

from src.pod.storage import store, TrajectoryStorage, DreamerStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.trainer.dreamertrainer import DreamerTrainer
from src.worldmodel.worldmodelinterface import WorldModelInterface
import jax.random as jr
import gymnasium as gym

import flax.linen as nn


class DreamerWrapper(gym.Wrapper):
    representation_model: ModelWrapper
    prev_state: jax.Array
    prev_belief: jax.Array

    def __init__(self, env, representation_model: ModelWrapper):
        super().__init__(env)
        self.representation_model = representation_model
        batch_shape = (Args().args.trajectory_length,
                       Args().args.num_envs)
        self.prev_belief = jnp.zeros((Args().args.num_envs, Args().args.belief_size))
        self.prev_state = jnp.zeros((Args().args.num_envs, Args().args.state_size))
        self.timestep = 0
        self.storage = DreamerStorage(observations=jnp.zeros(batch_shape + Shape()[0]),
                                      actions=jnp.zeros(batch_shape),
                                      rewards=jnp.zeros(batch_shape),
                                      beliefs=jnp.zeros(batch_shape + (Args().args.belief_size,)),
                                      states=jnp.zeros(batch_shape + (Args().args.state_size,)))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        batch = Args().args.num_envs
        self.prev_belief, self.prev_state, _, _, _, _ = (
            self.representation_model.forward(jnp.zeros((batch, self.representation_model.model.state_size)),
                                              jnp.zeros(batch),
                                              jnp.zeros((batch, self.representation_model.model.belief_size)),
                                              observation))
        self.timestep = 0
        return self.prev_state, info

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)
        belief, state, _, _, _, _ = self.representation_model.forward(self.prev_state, action,
                                                                      self.prev_belief, observation)
        self.storage = store(self.storage, self.timestep, observations=observation, actions=action, rewards=reward,
                             beliefs=self.prev_belief, states=self.prev_state)
        self.prev_belief = belief
        self.prev_state = state

        self.timestep += 1
        return self.prev_state, reward, term, trunc, info


class Dreamer(WorldModelInterface):
    def __init__(self, envs):
        self.num_agents = Args().args.num_agents
        self.belief_size = Args().args.belief_size
        self.state_size = Args().args.state_size
        self.embedding_size = 4 * 256
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
        transition_model = ModelWrapper(TransitionModel(self.belief_size, self.state_size,
                                                        self.action_size, self.hidden_size),
                                        "transition")
        observation_model = ModelWrapper(ObservationModel(self.belief_size, self.state_size,
                                                          self.embedding_size, self.observation_size),
                                         "observation")

        reward_model = ModelWrapper(RewardModel(self.belief_size, self.state_size, self.hidden_size), "reward")

        self.models = {"representation": representation_model,
                       "observation": observation_model,
                       "reward": reward_model,
                       "transition": transition_model}

        self.trainer = DreamerTrainer(representation_model.model, observation_model.model,
                                      reward_model.model)

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        print(self.prev_state.shape, action.shape, self.prev_belief.shape)
        self.prev_belief, self.prev_state, _, _ = self.models["transition"].forward(self.prev_state, action,
                                                                                    self.prev_belief)
        imagined_reward_logits = self.models["reward"].forward(self.prev_belief, self.prev_state)
        imagined_reward = jnp.argmax(nn.softmax(imagined_reward_logits), axis=-1)

        return self.prev_state, imagined_reward, jnp.zeros(imagined_reward.shape), jnp.zeros(imagined_reward.shape), {}

    def reset(self) -> (jax.Array, float, bool, bool, dict):
        key = Key().key(1)
        self.prev_belief = jr.choice(key, self.initial_beliefs, (self.num_agents,))
        self.prev_state = jr.choice(key, self.initial_states, (self.num_agents,))

        return self.prev_state, {}

    def save(self):
        for model in self.models.values():
            model.save()

    def load(self):
        for key, model in self.models.items():
            model.load(key)

    def update(self, data):
        observations, actions, rewards = data.observations, data.actions, data.rewards

        params = {key: model.params for key, model in self.models.items()}
        grads = self.trainer.train_step(observations, actions, rewards, params)

        self.initial_beliefs = data.beliefs.reshape(-1, self.belief_size)
        self.initial_states = data.states.reshape(-1, self.state_size)
        for key, model in self.models.items():
            if key == "transition":
                continue

            model.apply_grads(grads[key])

        new_params = {"params": self.models["representation"].params["params"]["transition_model"]}
        self.models["transition"].params = new_params

    def wrap_env(self, env):
        return DreamerWrapper(env, self.models["representation"])
