import argparse

from singleton_decorator import singleton


def parse_dqn(parser):
    parser.add_argument('--target_update_period', type=int, default=10, help='DQN: period of target network update')
    parser.add_argument('--start_steps', type=int, default=100, help='DQN: number of random steps before training')
    parser.add_argument('--update_every', type=int, default=50, help='DQN: number of steps between updates')
    parser.add_argument('--epsilon', type=float, default=0.2, help='DQN: epsilon for epsilon-greedy policy')
    parser.add_argument('--storage_size', type=int, default=10000, help='DQN: size of the replay buffer')
    return parser


def parse_ppo(parser):
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='PPO: lambda for GAE')
    parser.add_argument('--clip_threshold', type=float, default=0.1, help='PPO: clipping threshold')
    parser.add_argument('--regularization', type=float, default=0.01, help='PPO: regularization coefficient')
    parser.add_argument('--value_weight', type=float, default=0.5, help='PPO: coefficient of the value function')
    parser.add_argument('--max_action_repetitions', type=int, default=200, help='PPO: max amount of times the same '
                                                                                'action can be used in a row')
    return parser


def parse_simple(parser):
    parser.add_argument('--sim_trajectory_length', type=int, default=50,
                        help='Model_based: length of simulated trajectory')
    parser.add_argument('--pixel_reward', type=int, default=0.5, help='pixel loss weight')
    parser.add_argument('--kl_loss', type=int, default=0.8, help='KL divergence weight')
    parser.add_argument('--rewards', type=int, default=2, help='Simple: number of possible reward values')
    parser.add_argument('--softmax_loss_const', type=int, default=0.03, help='minimum softmax loss')
    parser.add_argument('--categorical_image', type=bool, default=True, help='Whether to use categorical distribution '
                                                                             'for images or mse')
    return parser


def parse_dreamer(parser):
    parser.add_argument('--belief_size', type=int, default=200, help='Dreamer: size of the deterministic belief')
    parser.add_argument('--state_size', type=int, default=200, help='Dreamer: size of the stochastic state')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='size of the hidden layers in reward and value networks')
    parser.add_argument('--min_std_dev', type=float, default=0.1,
                        help='Dreamer: minimum standard deviation of the policy')
    parser.add_argument('--gradient_steps', type=int, default=100, help='Dreamer: number of gradient steps')
    parser.add_argument('--loss_weights', type=float, nargs=3, default=[0.25, 0.5, 0.25], help='Dreamer: loss weights')
    parser.add_argument('--predict_dones', type=bool, default=True, help='whether to predict dones or just default '
                                                                         'to false')
    return parser


@singleton
class Args:
    def __init__(self):
        self._args = None

    @property
    def args(self):
        if self._args is None:
            self._args = self.parse_args()

        return self._args

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog="Flax RL", description="Flax RL for Atari and Mario")
        parser.add_argument('--algorithm', type=str, default='ppo',
                            help='the algorithm to use [simple, dreamer, dqn, ppo]')
        parser.add_argument('--env', type=str, default='breakout', help='the environment to use [breakout, mario]')
        parser.add_argument('--num-updates', type=int, default=4000, help='number of updates to train')
        parser.add_argument('--batch_size', type=int, default=50, help='the batch size for training')
        parser.add_argument('--num_envs', type=int, default=8, help='the number of parallel environments')
        parser.add_argument('--num_agents', type=int, default=8,
                            help='the number of parallel agents. Should equal num_envs, unless using model-based RL')
        parser.add_argument('--trajectory_length', type=int, default=400, help='the length of trajectory')
        parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='the learning rate of the optimizer')
        parser.add_argument('--discount_factor', type=float, default=0.99, help='the discount factor')

        parser.add_argument('--max_grad_norm', type=float, default=0.5,
                            help='the maximum norm for the gradient clipping')

        parser.add_argument('--seed', type=int, default=1, help='seed for reproducible benchmarks')
        parser.add_argument('--dropout', type=bool, default=True, help='whether to use dropout')
        parser.add_argument('--frame_stack', type=int, default=4, help='number of frames to stack')
        parser.add_argument('--frame_skip', type=int, default=4, help='number of frames to skip')
        parser.add_argument('--grayscale', type=bool, default=True, help='whether to use grayscale')
        parser.add_argument('--num_epochs', type=int, default=4,
                            help='number of epochs to train during each update')
        parser.add_argument('--model_updates', type=int, default=4,
                            help='number of updates of agent on the model in model-based RL')
        parser.add_argument('--hybrid_learning', type=bool, default=False, help='whether to train both with '
                                                                                'the model and with the environment')
        parser.add_argument('--initial_updates', type=int, default=100, help='number of updates of the model '
                                                                             'before starting to train the model')
        parser.add_argument('--sample_output', type=bool, default=False,
                            help='whether to sample the output of the model')
        parse_dqn(parser)
        parse_ppo(parser)
        parse_simple(parser)
        parse_dreamer(parser)
        args = parser.parse_args()
        args.num_minibatches = args.num_agents * args.trajectory_length // args.batch_size
        args.min_reward = 0 if args.env == 'breakout' else -6
        args.rewards = 2 if args.env == 'breakout' else 13
        return args

    @args.setter
    def args(self, value):
        self._args = value