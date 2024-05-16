import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

import gymnasium as gym
from gymnasium.wrappers import FrameStack, ResizeObservation
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace

from src.enviroment.observationwrappers import ReshapeObservation, FrameSkip
from src.enviroment.shape import Shape
from src.singletons.hyperparameters import Args


def make_envs(num_envs: int = 1, env_name: str = "breakout"):
    envs = gym.vector.SyncVectorEnv(*[make_env(env_name) for _ in range(Args().args.num_agents)])
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


def make_breakout() -> gym.Env:
    """
    Create the Breakout environment
    """
    env: gym.Env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = ResizeObservation(env, shape=(105, 80))
    env = optional_grayscale(env)
    env = FrameStack(env, num_stack=4)
    env = TimeLimit(env, max_episode_steps=Args().args.trajectory_length)
    env = ReshapeObservation(env)
    Shape.initialize(env)
    return env


def make_mario() -> gym.Env:
    env = gym_super_mario_bros.make("SuperMarioBros-v3", render_mode='rgb', apply_api_compatibility=True)

    env = FrameSkip(env, skip=4)
    env = optional_grayscale(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStack(env, num_stack=4)

    env = JoypadSpace(env, RIGHT_ONLY)
    return env


def optional_grayscale(env):
    if Args().args.grayscale:
        env = GrayScaleObservation(env, True)
    return env