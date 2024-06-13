from src.agent.agent import Agent
from src.modelfree import model_free_train_loop
from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.worldmodel.worldmodelinterface import WorldModelInterface
import jax.numpy as jnp

def sample_env(agent, envs):
    agent.store_trajectories = False
    model_free_train_loop(agent, envs)
    agent.store_trajectories = True
    return envs.storage


def interact_world_model(agent, world_model):
    init_state, _ = world_model.reset()
    print(jnp.mean(init_state))
    agent.receive_state(init_state)
    model_free_train_loop(agent, world_model, False, Args().args.sim_trajectory_length)


def update_agent(agent, world_model):
    for _ in range(Args().args.model_updates):
        interact_world_model(agent, world_model)


def model_based_train_loop(agent: Agent, world_model: WorldModelInterface, env, update_idx):
    storage = sample_env(agent, env)

    world_model.update(storage)

    saved_state = agent.last_transition()[3]
    update_agent(agent, world_model)

    agent.receive_state(saved_state)

