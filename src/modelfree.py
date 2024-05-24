import gymnasium as gym
import jax.numpy as jnp

from src.agent.agent import Agent
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.worldmodelinterface import WorldModelInterface


def model_free_train_loop(agent: Agent, envs: gym.Env):
    observation, _ = envs.reset()
    agent.receive_state(observation)
    writer = Writer().writer
    args = Args().args

    returns = jnp.zeros(observation.shape[0])
    for step in range(args.trajectory_length):
        StepTracker().increment(args.num_agents)
        reward, done, infos = interact(agent, envs)
        returns += reward

        # Only print when at least 1 env is done
        if not any(done):
            continue

        for ret in returns:
            writer.add_scalar("charts/episodic_return", ret, int(StepTracker()))


def interact(agent: Agent, environment: WorldModelInterface | gym.Env):
    action = agent.select_action()
    observation, reward, terminated, truncated, infos = environment.step(action)
    agent.receive_reward(reward)
    agent.receive_state(observation)
    dones = terminated | truncated
    agent.receive_done(dones)

    agent.update_policy()

    return reward, dones, infos
