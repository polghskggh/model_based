import os
import shutil
from typing import Optional

import gymnasium as gym

from src.agent.agent import Agent
from src.enviroment import make_env
from src.gpu import check_gpu
from src.modelbased import model_based_train_loop
from src.modelfree import model_free_train_loop
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances, ModelWriter
from src.worldmodel.simple import SimpleWorldModel
from src.worldmodel.worldmodelinterface import WorldModelInterface


def reset_checkpoints():
    ckpt_dir = hyperparameters["save_path"]
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


def main():
    reset_checkpoints()
    check_gpu()
    env = make_env()
    agent = Agent("dqn")
    # world_model = SimpleWorldModel(True)
    # run_n_episodes(100, agent, env, world_model)
    run_n_episodes(100, agent, env)
    agent.save()
    env.close()


def run_n_episodes(episodes: int, agent: Agent, env: gym.Env, world_model: Optional[WorldModelInterface] = None):
    writer_instances["reward"] = ModelWriter("reward", ["reward", "return"])

    for episode in range(episodes):
        run_experiment(agent, env, world_model)
        ModelWriter.save_all()                      # dynamically save gathered data
        ModelWriter.flush_all()


def run_experiment(agent: Agent, env: gym.Env, world_model: Optional[WorldModelInterface] = None):
    if world_model is None:
        model_free_train_loop(agent, env)
    else:
        model_based_train_loop(agent, world_model, env)


def pc_params():
    hyperparameters["max_episode_length"] = 60
    hyperparameters["ppo"]["number_of_trajectories"] = 1
    hyperparameters["ppo"]["trajectory_length"] = 60
    hyperparameters["ppo"]["batch_size"] = 30


if __name__ == '__main__':
    pc_params()
    main()
