import os
import shutil

import gymnasium as gym

from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.envinteraction import model_free_train_loop
from src.enviroment import make_env
from src.gpu import check_gpu
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances, ModelWriter


def reset_checkpoints():
    ckpt_dir = hyperparameters["save_path"]
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


def main():
    reset_checkpoints()
    check_gpu()
    env = make_env()
    agent = Agent("ppo")
    run_n_episodes(100, agent, env)
    env.close()


def run_n_episodes(episodes: int, agent: AgentInterface, env: gym.Env):
    results = writer_instances["reward"]
    for episode in range(episodes):
        run_experiment(agent, env, results)
        ModelWriter.save_all()                      # dynamically save gathered data
        ModelWriter.flush_all()
        agent.save()


def run_experiment(agent: AgentInterface, env: gym.Env, results: ModelWriter):
    model_free_train_loop(agent, env, results)


if __name__ == '__main__':
    main()
