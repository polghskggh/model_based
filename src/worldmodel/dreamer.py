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
from src.pod.trajectorystorage import TrajectoryStorage
from src.singletons.hyperparameters import Args
from src.trainer.dreamertrainer import DreamerTrainer
from src.worldmodel.worldmodelinterface import WorldModelInterface


class DreamerWrapper(gym.Wrapper):
    representation_model: ModelWrapper
    prev_state: jax.Array
    prev_belief: jax.Array

    def __init__(self, env: gym.Env, reperesentation_model: ModelWrapper):
        super.__init__(env)
        self.representation_model = reperesentation_model

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        observation, info = self.env.reset(**kwargs)
        self.prev_state, self.prev_belief, _, _ = (
            self.representation_model.forward(jnp.zeros(self.representation_model.model.state_size),
                                              jnp.eye(1, 4, 0),
                                              jnp.zeros(self.representation_model.model.belief_size),
                                              observation))
        return self.prev_state, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        observation, reward, term, trunc, info = self.env.step(action)
        self.prev_state, self.prev_belief, _, _ = self.representation_model.forward(self.prev_state, action,
                                                                                    observation)
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

        self.imagined_beliefs = []
        self.imagined_states = []

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
        imagined_belief, imagined_state, _, _ = self.models["transition"].forward(self.imagined_beliefs[-1],
                                                                                  self.imagined_states[-1], action)
        self.imagined_beliefs.append(imagined_belief.squeeze(dim=0))
        self.imagined_states.append(imagined_state.squeeze(dim=0))

        imagined_reward = self.models["reward"].forward(imagined_belief, imagined_state)

        return (imagined_belief, imagined_state), imagined_reward, 0, False, {}

    def reset(self) -> (jax.Array, float, bool, bool, dict):
        self.imagined_states = []
        self.imagined_beliefs = []

    def save(self):
        for model in self.models.values():
            model.save()

    def load(self):
        for key, model in self.models.items():
            model.load(key)

    def update(self, data: TrajectoryStorage):
        observations, actions, rewards, nonterminals = data

        params = {key: model.params for key, model in self.models.items()}
        new_params = self.trainer.train_step(observations, actions, rewards, nonterminals, params)

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
