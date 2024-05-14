import gymnasium as gym
import jax.numpy as jnp

from src.agent.agent import Agent
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances
from src.worldmodel.worldmodelinterface import WorldModelInterface


def model_free_train_loop(agent: Agent, env: gym.Env):
    observation, _ = env.reset()
    agent.receive_state(jnp.array(observation, dtype=jnp.float32))
    episode_return = 0

    for _ in range(hyperparameters["max_episode_length"]):
        reward, done = interact(agent, env)
        episode_return += reward
        writer_instances["reward"].add_data(reward, "reward")
        if done:
            writer_instances["reward"].add_data(episode_return, "return")
            return


def interact(agent: Agent, enviroment: WorldModelInterface | gym.Env, update: bool = True):
    action = agent.select_action()
    observation, reward, terminated, truncated, _ = enviroment.step(action)
    agent.receive_reward(reward)
    agent.receive_state(observation)

    done = terminated or truncated

    agent.receive_term(done)

    if update:
        agent.update_policy()

    return reward, done
