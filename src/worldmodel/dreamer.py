import jax

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
import jax.numpy as jnp


class Dreamer(WorldModelInterface):
    def __init__(self):
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
        for key, model in self.models.items():
            model.save(key)

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
        """ Infer belief over current state q(s_t|o≤t,a<t) from the history,
            return updated belief and posterior_state at time t
            returned shape: belief/state [belief/state_dim] (remove the time_dim)
        """
        # observation is obs.to(device), action.shape=[act_dim] (will add time dim inside this fn), belief.shape
        belief, _, _, _, posterior_state, _, _ = self.models["representation"].forward(state, action,
                                                                                       belief, observation)

        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
            dim=0)  # Remove time dimension from belief/state

        return belief, posterior_state


