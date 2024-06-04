import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

import gym
from gym.wrappers import FrameStack, ResizeObservation
from gym.wrappers import GrayScaleObservation
from gym.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace

from src.enviroment.wrappers import ReshapeObservation, FrameSkip
from src.enviroment.shape import Shape
from src.singletons.hyperparameters import Args


def make_envs():
    env_name = Args().args.env
    envs = gym.vector.SyncVectorEnv([lambda: make_env(env_name) for _ in range(Args().args.num_envs)])
    #envs = make_env(env_name)
    return envs


def make_env(env_name: str = "breakout") -> gym.Env:
    """
    Create the Breakout environment with the necessary wrappers
    :return: Breakout environment
    """
    if env_name == "mario":
        env = make_mario()
    else:
        env = make_breakout()
    return env


def apply_common_wrappers(env: gym.Env):

    return env


def make_breakout() -> gym.Env:
    """
    Create the Breakout environment
    """
    env: gym.Env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = ResizeObservation(env, shape=(105, 80))
    env = optional_grayscale(env)
    env = FrameStack(env, num_stack=Args().args.frame_stack)
    env = TimeLimit(env, max_episode_steps=Args().args.trajectory_length)
    env = ReshapeObservation(env)
    Shape.initialize(env)
    return env


def make_mario() -> gym.Env:
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = FrameSkip(env, skip=4)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = optional_grayscale(env)

    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, num_stack=Args().args.frame_stack)
    env = TimeLimit(env, max_episode_steps=Args().args.trajectory_length)
    env = ReshapeObservation(env)
    Shape.initialize(env)
    return env


def optional_grayscale(env):
    if Args().args.grayscale:
        env = GrayScaleObservation(env, True)
    return env