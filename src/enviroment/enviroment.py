import gymnasium as gym
from gymnasium.wrappers import FrameStack, ResizeObservation

from src.enviroment.observationreshape import ObservationReshape
from src.enviroment.shape import Shape


def make_env() -> gym.Env:
    """
    Create the Breakout environment with the necessary wrappers
    :return: Breakout environment
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = ResizeObservation(env, shape=(105, 80))
    env = FrameStack(env, num_stack=4)
    env = ObservationReshape(env)
    Shape(env)
    return env