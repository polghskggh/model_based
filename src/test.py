import gymnasium as gym
import jax.numpy as jnp
from gymnasium.wrappers import ResizeObservation
from jax import vmap
import jax.random as jr

from src.agent.agent import Agent
from src.enviroment import make_env, Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.strategy.modelstrategyfactory import model_strategy_factory
from src.pod.hyperparameters import hyperparameters
from src.trainer.saetrainer import SAETrainer
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


def test_ppo():
    hyperparameters["ppo"]["number_of_trajectories"] = 2
    hyperparameters["ppo"]["trajectory_length"] = 3
    agent = Agent("ppo")

    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    key, s1 = jr.split(key)
    key, s2 = jr.split(key)
    state = jr.uniform(s2, shape=(105, 80, 12), dtype=jnp.float32)
    state2 = jr.uniform(s1, shape=(105, 80, 12))
    state3 = jr.uniform(key, shape=(105, 80, 12))

    for e in range(100):
        agent.receive_state(state)
        action = agent.select_action()

        if action == 0:
            agent.receive_reward(1.0)
        elif action == 1:
            agent.receive_reward(-1.0)
        else:
            agent.receive_reward(0.0)

        agent.receive_state(state2)
        agent.receive_term(False)
        agent.update_policy()

        action = agent.select_action()

        if action == 0:
            agent.receive_reward(-1.0)
        elif action == 2:
            agent.receive_reward(1.0)
        else:
            agent.receive_reward(0.0)

        agent.receive_state(state3)
        agent.receive_term(False)
        agent.update_policy()

        action = agent.select_action()

        if action == 0:
            agent.receive_reward(-1.0)
        elif action == 3:
            agent.receive_reward(1.0)
        else:
            agent.receive_reward(0.0)

        agent.receive_state(state3)
        agent.receive_term(True)
        agent.update_policy()


def test_mario_env():
    env = make_env("mario")
    state, _ = env.reset()
    agent = Agent("ppo")
    for _ in range(10):
        agent.receive_state(state)
        action = agent.select_action()
        state, reward, done, _, _ = env.step(action)
        print("reward:", reward)
        if done:
            state, _ = env.reset()


def test():
    make_env()
    #test_mario_env()
    test_ppo()
    #test_stochastic_autoencoder()


if __name__ == '__main__':
    test()