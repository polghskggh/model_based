import gymnasium as gym
import jax.numpy as jnp
from jax import lax

from src.agent.agent import Agent
from src.enviroment import Shape
from src.modelfree import interact, model_free_train_loop
from src.pod.storage import TransitionStorage, store
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer
from src.worldmodel.worldmodelinterface import WorldModelInterface


def sample_env(agent, envs):
    model_free_train_loop(agent, envs)
    return envs.storage


def model_based_train_loop(agent: Agent, world_model: WorldModelInterface, env: gym.Env):
    agent.store_trajectories = False
    storage = sample_env(agent, world_model.wrap_env(env))
    world_model.update(storage)
    agent.store_trajectories = True

    for update in range(Args().args.model_updates):
        model_free_train_loop(agent, world_model, False)


