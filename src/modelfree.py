import gymnasium as gym
import jax.numpy as jnp

from src.agent.agent import Agent
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.worldmodelinterface import WorldModelInterface


def write_returns(infos):
    writer = Writer().writer
    for info in infos["final_info"]:
        if info is not None:
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], int(StepTracker()))
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], int(StepTracker()))


def model_free_train_loop(agent: Agent, envs: gym.Env | WorldModelInterface, increment: bool = True,
                          steps=Args().args.trajectory_length):
    for step in range(steps):
        reward, done, infos = interact(agent, envs)

        if increment:
            StepTracker().increment(Args().args.num_envs)

        # Only print when at least 1 env is done
        if increment and jnp.any(done):
            write_returns(infos)


def interact(agent: Agent, environment: WorldModelInterface | gym.Env):
    action = agent.select_action()
    observation, reward, terminated, truncated, infos = environment.step(action)
    agent.receive_reward(reward)
    agent.receive_state(observation)
    dones = terminated | truncated
    agent.receive_done(dones)

    agent.update_policy()

    return reward, dones, infos
