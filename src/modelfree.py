import gymnasium as gym
import jax.numpy as jnp

from src.agent.agent import Agent
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.worldmodelinterface import WorldModelInterface


def model_free_train_loop(agent: Agent, env: gym.Env):
    observation, _ = env.reset()
    agent.receive_state(jnp.array(observation, dtype=jnp.float32))
    writer = Writer().writer
    args = Args().args

    returns = jnp.zeros(args.num_agents)
    for step in range(args.trajectory_length):
        StepTracker().increment(args.num_agents)
        reward, done, infos = interact(agent, env)
        returns += reward

        # Only print when at least 1 env is done
        if not any(done):
            continue

        for ret in returns:
            writer.add_scalar("charts/episodic_return", ret, int(StepTracker()))


def interact(agent: Agent, environment: WorldModelInterface | gym.Env, update: bool = True):
    action = agent.select_action()
    observation, reward, terminated, truncated, infos = environment.step(action)
    agent.receive_reward(reward)
    agent.receive_state(observation)
    dones = terminated | truncated
    agent.receive_done(dones)

    if update:
        agent.update_policy()

    return reward, dones, infos
