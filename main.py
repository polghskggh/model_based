import os
import shutil

import gymnasium as gym

from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.modelbased import model_based_train_loop
from src.modelfree import model_free_train_loop
from src.enviroment import make_env
from src.gpu import check_gpu
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances, ModelWriter
from src.worldmodel.deterministicsimple import DeterministicSimple
import jax

from src.worldmodel.worldmodelinterface import WorldModelInterface


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
    agent.save()
    env.close()


def run_n_episodes(episodes: int, agent: AgentInterface, env: gym.Env):
    writer_instances["reward"] = ModelWriter("reward", ["reward", "return"])
    world_model = DeterministicSimple(env)

    for episode in range(episodes):
        run_experiment(agent, env, world_model)
        ModelWriter.save_all()                      # dynamically save gathered data
        ModelWriter.flush_all()


def run_experiment(agent: AgentInterface, env: gym.Env, world_model: WorldModelInterface):
    #model_based_train_loop(agent, world_model, env)
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        model_free_train_loop(agent, env)


if __name__ == '__main__':
    main()
