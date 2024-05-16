import argparse

from singleton_decorator import singleton

hyperparameters = {
    'max_episode_length': 1024,
    'grayscale': False,
    'dropout': True,
    'max_grad_norm': 0.5,
    'simple': {
        'deterministic_lr': 0.0001,
        'stochastic_lr': 0.0001,
        'lstm_lr': 0.0001,
        'kl_loss_weight': 2,
        'batch_size': 5,
        'data_size': 10,
        'rollout_length': 3,
        'kl_loss': 0.8,
        'pixel_reward': 0.5,
        'parallel_agents': 2,
        'agent_updates': 4
    },
    'dqn': {
        'batch_size': 100,
        'batches_per_update': 5,
        'discount_factor': 0.95,
        'target_update_period': 10,
        'start_steps': 100,
        'update_every': 50,
        'epsilon': 0.2
    },
    'ppo': {
        'batch_size': 300,
        'number_of_trajectories': 10,
        'discount_factor': 0.99,
        'lambda': 0.95,
        'clip_threshold': 0.2,
        'trajectory_length': 1024,
        'regularization': 0.3,
        'actor_lr': 1e-4,
    },
    'dreamer': {
        'belief_size': 200,
        'state_size': 30,
        'embedding_size': 30,
        'hidden_size': 200,
        'min_std_dev': 0.1,
        'gradient_steps': 100,
        'batch_size': 100,
        'loss_weights': (0.25, 0.5, 0.25),
    },
    'rng': {
        'dropout': 0,
        'normal': 0,
        'carry': 0,
        'params': 0,
        'action': 0,
    },
    'frame_stack': 4,
    'save_path': '/tmp/flax_ckpt',
}


def parse_dqn(subparser):
    parser = subparser.add_parser('dqn', help='DQN hyperparameters')
    parser.add_argument('--batches_per_update', type=int, default=10, help='number of batches used at each update')
    parser.add_argument('--target_update_period', type=int, default=10, help='period of target network update')
    parser.add_argument('--start_steps', type=int, default=100, help='number of random steps before training')
    parser.add_argument('--update_every', type=int, default=50, help='number of steps between updates')
    parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon for epsilon-greedy policy')
    parser.add_argument('--storage_size', type=int, default=1e5, help='size of the replay buffer')
    return parser


def parse_ppo(subparser):
    parser = subparser.add_parser('ppo', help='PPO hyperparameters')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='lambda for GAE')
    parser.add_argument('--clip-threshold', type=float, default=0.1, help='clipping threshold')
    parser.add_argument('--regularization', type=float, default=0.01, help='regularization coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient of the value function')


def parse_simple(subparser):
    parser = subparser.add_parser('simple', help='Simple hyperparameters')
    return parser


def parse_dreamer(subparser):
    parser = subparser.add_parser('dreamer', help='Dreamer hyperparameters')
    parser.add_argument('--belief-size', type=int, default=200, help='size of the deterministic belief')
    parser.add_argument('--state-size', type=int, default=200, help='size of the stochastic state')
    parser.add_argument('--embedding-size', type=int, default=30, help='size of the encoder embedding')
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='size of the hidden layers in reward and value networks')
    parser.add_argument('--min-std-dev', type=float, default=0.1, help='minimum standard deviation of the policy')
    parser.add_argument('--gradient-steps', type=int, default=100, help='number of gradient steps')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
    parser.add_argument('--loss-weights', type=float, nargs=3, default=[0.25, 0.5, 0.25])
    return parser


@singleton
class Args:
    args = None

    def __call__(self):
        if self.args is None:
            self.args = self.parse_args()

        return self.args

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog="Flax RL", description="Flax RL for Atari and Mario")

        parser.add_argument('--env', type=str, default='breakout', help='the environment to use [breakout, mario]')
        parser.add_argument('--num-episodes', type=int, default=1e5, help='number of episodes to train')
        parser.add_argument('--num-agents', type=int, default=8, help='the number of parallel agents')
        parser.add_argument('--trajectory-len', type=int, default=128, help='the length of trajectory')
        parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='the learning rate of the optimizer')

        parser.add_argument('--discount', type=float, default=0.99, help='the discount factor')
        parser.add_argument('--max-grad-norm', type=float, default=0.5,
                            help='the maximum norm for the gradient clipping')

        parser.add_argument('--seed', type=int, default=1, help='seed for reproducible benchmarks')
        parser.add_argument('--dropout', type=bool, default=False, help='whether to use dropout')
        subparsers = parser.add_subparsers(help='hyperparameters for different algorithms', dest='algorithm')
        parse_dqn(subparsers)
        parse_ppo(subparsers)
        parse_simple(subparsers)
        parse_dreamer(subparsers)

        args = parser.parse_args()
        return args