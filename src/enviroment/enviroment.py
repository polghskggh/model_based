import gymnasium as gym
from gymnasium.wrappers import FrameStack, ResizeObservation

from src.enviroment.observationreshape import ObservationReshape
from src.enviroment.onehotaction import OneHotAction


def make_env() -> gym.Env:
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = ResizeObservation(env, shape=(105, 80))
    env = FrameStack(env, num_stack=4)
    env = ObservationReshape(env)
    env = OneHotAction(env)
    return env
