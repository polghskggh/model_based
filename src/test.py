from jax import vmap

from src.agent.acstrategy import shapes
from src.enviroment import make_env
from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.modelwrapper import ModelWrapper
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation

import jax.numpy as jnp

from src.models.strategy.modelstrategyfactory import model_strategy_factory
from src.utils.inttoonehot import image_to_onehot
from src.utils.tiling import tile_image


def setup():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = ResizeObservation(env, shape=(105, 80))
    observation, _ = env.reset()
    autoencoder = ModelWrapper(AutoEncoder(*shapes["atari-ddpg"]), "autoencoder")
    stack, action = model_strategy_factory("autoencoder").init_params(autoencoder._model)
    observation = image_to_onehot(tile_image(observation))
    return env, autoencoder, stack, action, observation


def test_autoencoder():
    """
    Test the autoencoder on dummy input
    """
    env, autoencoder, stack, action, observation = setup()
    for i in range(100):
        grads = autoencoder.train_step(observation, stack, action)
        autoencoder.apply_grads(grads)
        print(jnp.max(observation), jnp.max(autoencoder.forward(stack, action)))

    env.close()


def test_autoencoder_actions():
    """
    Test the autoencoder on dummy input with different actions
    Does the autoencoder learn to differentiate between actions?
    """
    env, autoencoder, stack, action, observation = setup()
    action = jnp.array([1, 0, 0, 0])

    observation2 = jnp.array(observation, copy=True)
    action2 = jnp.array([0, 0, 0, 1])
    observation2.at[0, 0].set(jnp.eye(256)[5])
    for i in range(10):
        grads = autoencoder.train_step(observation, stack, action)
        autoencoder.apply_grads(grads)

        grads = autoencoder.train_step(observation2, stack, action2)
        autoencoder.apply_grads(grads)

    env.close()
    print(autoencoder.forward(stack, action)[0, 0, 0, 5] - autoencoder.forward(stack, action2)[0, 0, 0, 5])


def test():
    test_autoencoder_actions()


if __name__ == '__main__':
    test()
