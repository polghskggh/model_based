import os
import shutil
import time

from typing import Optional

import jax.random as jr
import gymnasium as gym
from termcolor import colored
from tqdm import tqdm

from src.agent.agent import Agent
from src.enviroment import make_env
from src.gpu import check_gpu
from src.modelbased import model_based_train_loop
from src.modelfree import model_free_train_loop
from src.pod.hyperparameters import hyperparameters, Args

from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.dreamer import Dreamer
from src.worldmodel.simple import SimpleWorldModel
from src.worldmodel.worldmodelinterface import WorldModelInterface


def reset_checkpoints():
    ckpt_dir = hyperparameters["save_path"]
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)


def world_model_factory():
    args = Args()
    match args.algorithm:
        case "simple":
            return SimpleWorldModel(True)
        case "dreamer":
            return Dreamer()
        case _:
            return None


def main():
    reset_checkpoints()
    check_gpu()
    envs = make_env()
    agent = Agent()
    world_model = world_model_factory()
    # world_model = SimpleWorldModel(True)
    # run_n_episodes(100, agent, env, world_model)
    run_experiment(100, agent, envs)


def run_experiment(updates: int, agent: Agent, envs: gym.Env, world_model: Optional[WorldModelInterface] = None):
    start_time = time.time()
    writer = Writer().writer
    initial_observation, _ = envs.reset(seed=Args().seed)
    agent.receive_state(initial_observation)

    try:
        for update in tqdm(range(1, updates + 1)):
            run_episode(agent, world_model, envs)
            global_step = StepTracker()
            writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(),
                              global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print(colored('Training complete!', 'green'))
    except KeyboardInterrupt:
        print(colored('Training interrupted!', 'red'))
    finally:
        envs.close()
        writer.close()
        agent.save()
        if world_model is not None:
            world_model.save()


def run_episode(agent: Agent, world_model: Optional[WorldModelInterface], envs: gym.Env):
    if world_model is None:
        model_free_train_loop(agent, envs)
    else:
        model_based_train_loop(agent, world_model, envs)


def pc_params():
    hyperparameters["max_episode_length"] = 60
    hyperparameters["ppo"]["number_of_trajectories"] = 1
    hyperparameters["ppo"]["trajectory_length"] = 60
    hyperparameters["ppo"]["batch_size"] = 30


if __name__ == '__main__':
    main()
