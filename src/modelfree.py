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

    for step in range(args.trajectory_length):
        StepTracker().increment(args.num_agents)
        reward, done, infos = interact(agent, env)

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue

            writer.add_scalar("charts/episodic_return", info["episode"]["r"], int(StepTracker()))
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], int(StepTracker()))


def interact(agent: Agent, environment: WorldModelInterface | gym.Env, update: bool = True):
    action = agent.select_action()
    observation, reward, terminated, truncated, infos = environment.step(action)
    agent.receive_reward(reward)
    agent.receive_state(observation)
    agent.receive_done(terminated or truncated)

    if update:
        agent.update_policy()

    return reward, terminated or truncated, infos
