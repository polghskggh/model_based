import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

import gymnasium as gym

from gymnasium.wrappers import FrameStack, ResizeObservation, RecordEpisodeStatistics
from gymnasium.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace

from src.enviroment.wrappers import ReshapeObservation, FrameSkip, CompatibilityWrapper, LimitActions
from src.enviroment.shape import Shape
from src.singletons.hyperparameters import Args


def make_envs():
    env_name = Args().args.env
    envs = gym.vector.SyncVectorEnv([lambda: make_env(env_name) for _ in range(Args().args.num_envs)])
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
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, (84, 84))
    env = optional_grayscale(env)
    if Args().args.algorithm != "dreamer":
        env = FrameStack(env, Args().args.frame_stack)
        env = ReshapeObservation(env)
    Shape.initialize(env)
    return env


def make_breakout() -> gym.Env:
    """
    Create the Breakout environment
    """
    env: gym.Env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=4)
    env = apply_common_wrappers(env)
    return env


def make_mario() -> gym.Env:
    env = gym_super_mario_bros.make("SuperMarioBros-v3", render_mode="rgb_array", apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = CompatibilityWrapper(env)
    env = LimitActions(env)
    env = FrameSkip(env, 4)
    env = apply_common_wrappers(env)
    return env


def optional_grayscale(env):
    if Args().args.grayscale:
        env = GrayScaleObservation(env, True)
    return env