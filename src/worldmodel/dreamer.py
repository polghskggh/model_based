from typing import Tuple

import gym
import jax
import jax.numpy as jnp

from gym.core import ActType, ObsType

from src.enviroment import Shape
from src.models.dreamer.observation import ObservationModel
from src.models.dreamer.representation import RepresentationModel
from src.models.dreamer.reward import RewardModel
from src.models.dreamer.transition import TransitionModel
from src.models.modelwrapper import ModelWrapper

from src.pod.storage import store, TrajectoryStorage
from src.singletons.hyperparameters import Args
from src.trainer.dreamertrainer import DreamerTrainer
from src.worldmodel.worldmodelinterface import WorldModelInterface


class DreamerWrapper(gym.Wrapper):
    representation_model: ModelWrapper
    prev_state: jax.Array
    prev_belief: jax.Array

    def __init__(self, env: gym.Env, reperesentation_model: ModelWrapper):
        super().__init__(env)
        self.representation_model = reperesentation_model
        batch_shape = (Args().args.trajectory_length,
                       Args().args.num_agents)
        self.prev_belief = jnp.zeros((Args().args.num_agents, Args().args.belief_size))
        self.prev_state = jnp.zeros((Args().args.num_agents, Args().args.state_size))
        self.timestep = 0
        self.storage = TrajectoryStorage(observations=jnp.zeros(batch_shape + Shape()[0]),
                                         actions=jnp.zeros(batch_shape),
                                         rewards=jnp.zeros(batch_shape))

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        observation, info = self.env.reset(**kwargs)
        batch = Args().args.num_agents
        self.prev_state, self.prev_belief, _, _ = (
            self.representation_model.forward(jnp.zeros((batch, self.representation_model.model.state_size)),
                                              jnp.zeros(batch),
                                              jnp.zeros((batch, self.representation_model.model.belief_size)),
                                              observation))
        self.timestep = 0
        return self.prev_state, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        observation, reward, term, trunc, info = self.env.step(action)
        self.prev_state, self.prev_belief, _, _ = self.representation_model.forward(self.prev_state, action,
                                                                                    self.prev_belief, observation)
        self.storage = store(self.storage, self.timestep, observations=observation, actions=action, rewards=reward)
        self.timestep += 1
        return self.prev_state, reward, term, trunc, info


class Dreamer(WorldModelInterface):
    def __init__(self, envs):
        self.batch_size = Args().args.batch_size
        self.belief_size = Args().args.belief_size
        self.state_size = Args().args.state_size
        self.embedding_size = Args().args.embedding_size
        self.hidden_size = Args().args.hidden_size
        self.observation_size = Shape()[0]
        self.action_size = Shape()[1]

        self.prev_belief = jnp.zeros(self.batch_size, self.belief_size)
        self.prev_state = jnp.zeros(self.batch_size, self.state_size)

        transition_model = ModelWrapper(TransitionModel(self.belief_size, self.state_size, self.action_size,
                                                        self.hidden_size), "transition")
        # Initialise model parameters randomly
        representation_model = ModelWrapper(RepresentationModel(self.belief_size, self.state_size,
                                                                self.action_size, self.hidden_size,
                                                                self.embedding_size, Shape()[0]),
                                            "representation")
        observation_model = ModelWrapper(ObservationModel(self.belief_size, self.state_size,
                                                          self.embedding_size, self.observation_size),
                                         "observation")

        reward_model = ModelWrapper(RewardModel(self.belief_size, self.state_size, self.hidden_size), "reward")

        self.models = {"representation": representation_model,
                       "transition": transition_model,
                       "observation": observation_model,
                       "reward": reward_model}

        self.trainer = DreamerTrainer(representation_model.model, observation_model.model,
                                      reward_model.model)

    def step(self, action) -> (jax.Array, float, bool, bool, dict):
        self.prev_belief, self.prev_state, _, _ = self.models["transition"].forward(self.prev_beliefs,
                                                                                    action, self.prev_state)

        imagined_reward = self.models["reward"].forward(self.prev_belief, self.prev_state)

        return self.prev_state, imagined_reward, 0, False, {}

    def reset(self) -> (jax.Array, float, bool, bool, dict):
        self.prev_belief = jnp.zeros(self.batch_size, self.belief_size)
        self.prev_state = self.models["transition"].forward(self.prev_belief,
                                                            jnp.zeros(self.batch_size, self.state_size),
                                                            jnp.zeros(self.batch_size, self.state_size))
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
        new_params = self.trainer.train_step(observations, actions, rewards, params)

        for key, model in self.models.items():
            model.params = new_params[key]

    def infer_state(self, observation, action, belief=None, state=None):
        # observation is obs.to(device), action.shape=[act_dim] (will add time dim inside this fn), belief.shape
        belief, _, _, _, posterior_state, _, _ = self.models["representation"].forward(state, action,
                                                                                       belief, observation)

        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
            dim=0)  # Remove time dimension from belief/state

        return belief, posterior_state

    def wrap_env(self, env):
        return DreamerWrapper(env, self.models["representation"])
