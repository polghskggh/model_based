import jax

from src.enviroment import Shape
from src.models.dreamer.observation import ObservationModel
from src.models.dreamer.representation import RepresentationModel
from src.models.dreamer.reward import RewardModel
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters
from src.pod.trajectorystorage import TrajectoryStorage
from src.worldmodel.worldmodelinterface import WorldModelInterface
import jax.numpy as jnp



def cal_returns(reward, value, bootstrap, pcont, lambda_):
    """
    Calculate the target value, following equation (5-6) in Dreamer
    :param reward, value: imagined rewards and values, dim=[horizon, (chuck-1)*batch, reward/value_shape]
    :param bootstrap: the last predicted value, dim=[(chuck-1)*batch, 1(value_dim)]
    :param pcont: gamma
    :param lambda_: lambda
    :return: the target value, dim=[horizon, (chuck-1)*batch, value_shape]
    """
    assert list(reward.shape) == list(value.shape), "The shape of reward and value should be similar"

    next_value = torch.cat((value[1:], bootstrap[None]), 0)  # bootstrap[None] is used to extend additional dim
    inputs = reward + next_value * (1 - lambda_)  # dim=[horizon, (chuck-1)*B, 1]
    outputs = []
    last = bootstrap

    for t in reversed(range(reward.shape[0])):  # for t in horizon
        inp = inputs[t]
        last = inp + pcont[t] * lambda_ * last
        outputs.append(last)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


class Dreamer(WorldModelInterface):
    def __init__(self):
        super().__init__()
        self.batch_size = hyperparameters["dreamer"]["batch_size"]
        self.belief_size = hyperparameters["dreamer"]["belief_size"]
        self.state_size = hyperparameters["dreamer"]["state_size"]
        self.embedding_size = hyperparameters["dreamer"]["embedding_size"]
        self.hidden_size = hyperparameters["dreamer"]["hidden_size"]
        self.observation_size = Shape()[0]
        self.action_size = Shape()[1]

        # Initialise model parameters randomly
        self.representation_model = ModelWrapper(RepresentationModel(self.belief_size, self.state_size,
                                                                     self.action_size, self.hidden_size,
                                                                     self.embedding_size), "representation")
        self.observation_model = ModelWrapper(ObservationModel(self.belief_size, self.state_size,
                                                               self.embedding_size, self.observation_size),
                                              "observation")

        self.reward_model = ModelWrapper(RewardModel(self.belief_size, self.state_size, self.hidden_size), "reward")

    def process_im(self, image):
        # Resize, put channel first, convert it to a tensor, centre it to [-0.5, 0.5] and add batch dimenstion.

        def preprocess_observation_(observation, bit_depth):
            # Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
            observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(
                0.5)  # Quantise to given bit depth and centre
            observation.add_(torch.rand_like(observation).div_(
                2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

        image = torch.tensor(cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                             dtype=torch.float32)  # Resize and put channel first

        preprocess_observation_(image, self.args.bit_depth)
        return image.unsqueeze(dim=0)


    def _latent_imagination(self, beliefs, posterior_states, with_logprob=False):
        # Rollout to generate imagined trajectories

        chunk_size, batch_size, _ = list(posterior_states.size())  # flatten the tensor
        flatten_size = chunk_size * batch_size

        posterior_states = posterior_states.detach().reshape(flatten_size, -1)
        beliefs = beliefs.detach().reshape(flatten_size, -1)

        imag_beliefs, imag_states, imag_ac_logps = [beliefs], [posterior_states], []

        for i in range(self.args.planning_horizon):
            imag_action, imag_ac_logp = self.actor_model(
                imag_beliefs[-1].detach(),
                imag_states[-1].detach(),
                deterministic=False,
                with_logprob=with_logprob,
            )
            imag_action = imag_action.unsqueeze(dim=0)  # add time dim

            imag_belief, imag_state, _, _ = self.transition_model(imag_states[-1], imag_action, imag_beliefs[-1])
            imag_beliefs.append(imag_belief.squeeze(dim=0))
            imag_states.append(imag_state.squeeze(dim=0))

            if with_logprob:
                imag_ac_logps.append(imag_ac_logp.squeeze(dim=0))

        imag_beliefs = torch.stack(imag_beliefs, dim=0).to(
            self.args.device)  # shape [horizon+1, (chuck-1)*batch, belief_size]
        imag_states = torch.stack(imag_states, dim=0).to(self.args.device)

        if with_logprob:
            imag_ac_logps = torch.stack(imag_ac_logps, dim=0).to(self.args.device)  # shape [horizon, (chuck-1)*batch]

        return imag_beliefs, imag_states, imag_ac_logps if with_logprob else None

    def update(self, data: TrajectoryStorage):
        # get state and belief of samples
        observations, actions, rewards, nonterminals = data



    def infer_state(self, observation, action, belief=None, state=None):
        """ Infer belief over current state q(s_t|o≤t,a<t) from the history,
            return updated belief and posterior_state at time t
            returned shape: belief/state [belief/state_dim] (remove the time_dim)
        """
        # observation is obs.to(device), action.shape=[act_dim] (will add time dim inside this fn), belief.shape
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            state,
            action.unsqueeze(dim=0),
            belief,
            self.encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension

        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
            dim=0)  # Remove time dimension from belief/state

        return belief, posterior_state

    def select_action(self, state, deterministic=False):
        # get action with the inputs get from fn: infer_state; return a numpy with shape [batch, act_size]
        belief, posterior_state = state
        action, _ = self.actor_model(belief, posterior_state, deterministic=deterministic, with_logprob=False)

        if not deterministic and not self.args.with_logprob:  ## add exploration noise
            action = Normal(action, self.args.expl_amount).rsample()
            action = torch.clamp(action, -1, 1)
        return action  # tensor
