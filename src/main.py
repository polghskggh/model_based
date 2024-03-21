import gym
import jax
import numpy as np

from src.agent.acstrategy import shapes
from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.enviroment import make_env
from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.modelwrapper import ModelWrapper
from src.resultwriter.modelwriter import writer_instances, ModelWriter
from jax import devices


def main():
    check_gpu()
    env = make_env()
    agent = Agent("atari-ddpg")
    run_n_episodes(100, agent, env)
    env.close()


def run_n_episodes(episodes: int, agent: AgentInterface, env: gym.Env):
    results = writer_instances["reward"]
    for episode in range(episodes):
        run_experiment(agent, env, results)
        results.save_episode()                      # dynamically save gathered data
        results.flush_all()


def run_experiment(agent: AgentInterface, env: gym.Env, results: ModelWriter):
    observation_old, _ = env.reset()
    autoencoder = ModelWrapper(AutoEncoder(shapes["atari-ddpg"][0], 1), "autoencoder")
    print(autoencoder)
    for _ in range(1000):
        action = env.action_space.sample()
        observation_new, reward, terminated, truncated, _ = env.step(action)

        observation_old = observation_new

        if terminated or truncated:
            return


def check_gpu():
    try:
        gpu_devices = devices('gpu')
        print("GPU is available.")
        for gpu in gpu_devices:
            print(gpu)
    except RuntimeError:
        print("No GPU available.")


if __name__ == '__main__':
    main()
