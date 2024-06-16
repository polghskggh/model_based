import time
from typing import Optional

import gymnasium as gym
from termcolor import colored
from tqdm import tqdm

from src.agent.agent import Agent
from src.enviroment import make_envs, Shape
from src.gpu import check_gpu
from src.modelbased import model_based_train_loop, initial_training
from src.modelfree import model_free_train_loop
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.dreamer import Dreamer
from src.worldmodel.simple import SimpleWorldModel
from src.worldmodel.worldmodelinterface import WorldModelInterface


def world_model_factory(envs):
    args = Args().args
    match args.algorithm:
        case "simple":
            world_model = SimpleWorldModel(True)
            return world_model, world_model.wrap_env(envs)
        case "dreamer":
            world_model = Dreamer()
            return world_model, world_model.wrap_env(envs)
        case _:
            return None, envs


def main():
    check_gpu()
    envs = make_envs()
    agent = Agent(Args().args.algorithm)
    world_model, envs = world_model_factory(envs)
    if world_model is not None:
        world_model = initial_training(envs, world_model)
    run_experiment(agent, envs, world_model)


def run_experiment(agent: Agent, envs: gym.Env, world_model: Optional[WorldModelInterface] = None):
    start_time = time.time()
    writer = Writer().writer

    initial_observation, _ = envs.reset(seed=Args().args.seed)
    agent.receive_state(initial_observation)

    try:
        for update in tqdm(range(1, Args().args.num_updates + 1)):
            run_episode(agent, world_model, envs, update)
            global_step = int(StepTracker())
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print(colored('Training complete!', 'green'))
    except KeyboardInterrupt:
        print(colored('Training interrupted!', 'red'))
    finally:
        envs.close()
        writer.close()


def run_episode(agent: Agent, world_model: Optional[WorldModelInterface], envs: gym.Env, update: int):
    if world_model is None:
        model_free_train_loop(agent, envs)
    else:
        model_based_train_loop(agent, world_model, envs, update)


if __name__ == '__main__':
    main()
    # presentation
    # slides 10 - 15
    #