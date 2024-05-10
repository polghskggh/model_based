import gymnasium as gym
import jax.numpy as jnp
from gymnasium.wrappers import ResizeObservation
from jax import vmap

from src.agent.agent import Agent
from src.enviroment import make_env, Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.strategy.modelstrategyfactory import model_strategy_factory
from src.models.trainer.saetrainer import SAETrainer
from src.utils.tiling import tile_image


def gen_autoencoder(stochastic: bool):
    """
    Generate the autoencoder
    :return: Wrapper for the autoencoder
    """
    if stochastic:
        return ModelWrapper(StochasticAutoencoder(*Shape()), "stochastic_autoencoder")

    return ModelWrapper(AutoEncoder(*Shape()), "autoencoder")


def setup(stochastic: bool = False):
    """
    Setup the environment and the autoencoder
    """
    e = make_env()
    e.close()
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = ResizeObservation(env, shape=(105, 80))

    next_frame, _ = env.reset()
    autoencoder = gen_autoencoder(stochastic)
    stack, action = model_strategy_factory("autoencoder").init_params(autoencoder.model)
    return env, autoencoder, stack, action, next_frame


def test_deterministic_autoencoder():
    """
    Test the deterministic autoencoder on dummy input
    """
    env, autoencoder, stack, action, _, observation = setup()
    for i in range(100):
        grads = autoencoder.train_step(observation, stack, action)
        autoencoder.apply_grads(grads)
        print(jnp.max(observation), jnp.max(autoencoder.forward(stack, action)))

    env.close()


def test_deterministic_autoencoder_actions():
    """
    Test the deterministic autoencoder on dummy input with different actions
    Does the deterministic autoencoder learn to differentiate between actions?
    """
    env, autoencoder, stack, action, _, observation = setup()
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


def test_deterministic_on_loaded_data():
    """
    Test the autoencoder on loaded data
    """
    autoencoder = gen_autoencoder()
    files = jnp.load("../data/data_0.npz")
    stack, action, observation = (jnp.asarray(files["frame_stack"]),
                                  jnp.asarray(files["action"]),
                                  jnp.asarray(files["next_frame"]))

    observation = vmap(tile_image)(observation)
    for _ in range(10):
        grads = autoencoder.train_step(observation[:10], stack[:10], action[:10])
        autoencoder.apply_grads(grads)

    print("done")


def test_stochastic_autoencoder():
    """
    Test the autoencoder on dummy input
    """
    env, autoencoder, stack, action, next_frame, observation = setup(stochastic=True)
    trainer = SAETrainer(autoencoder.model)
    for i in range(5):
        params = trainer.train_step(autoencoder.params, stack, action, next_frame)
        autoencoder.params = params
    env.close()


def test_PPO():
    agent = Agent("ppo")
    state = jnp.zeros((1, 105, 80, 12))
    advantage = jnp.zeros((1, 1))
    agent.receive_state(state)
    action = agent.select_action()
    agent.receive_reward(0)
    agent.receive_state(state)
    agent.receive_term(True)
    for _ in range(10):
        print("updating")
        agent.update_policy()


def test():
    make_env()
    test_PPO()
    #test_stochastic_autoencoder()


if __name__ == '__main__':
    test()
