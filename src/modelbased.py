from jax import numpy as jnp

from src.agent.agent import Agent
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment import Shape
from src.modelfree import model_free_train_loop
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.worldmodel.worldmodelinterface import WorldModelInterface
import jax.random as jr

def sample_env(agent, envs):
    agent.store_trajectories = False
    model_free_train_loop(agent, envs)
    agent.store_trajectories = True
    return envs.storage


def update_agent(agent, world_model):
    saved_state = agent.last_transition()[3]
    for _ in range(Args().args.model_updates):
        init_state, _ = world_model.reset()
        agent.receive_state(init_state)
        model_free_train_loop(agent, world_model, False, Args().args.sim_trajectory_length)

    agent.receive_state(saved_state)


def model_based_train_loop(agent: Agent, world_model: WorldModelInterface, env, update_idx):
    storage = sample_env(agent, env)
    world_model.update(storage)
    update_agent(agent, world_model)


def initial_training(envs, world_model):
    _, _ = envs.reset()
    for _ in range(Args().args.initial_updates):
        storage = sample_env(Agent("random"), envs)
        world_model.update(storage)
