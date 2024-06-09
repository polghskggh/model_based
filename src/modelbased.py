from src.agent.agent import Agent
from src.modelfree import model_free_train_loop
from src.singletons.hyperparameters import Args
from src.worldmodel.worldmodelinterface import WorldModelInterface


def sample_env(agent, envs):
    agent.store_trajectories = False
    model_free_train_loop(agent, envs)
    agent.store_trajectories = True
    return envs.storage


def interact_world_model(agent, world_model):
    init_state, _ = world_model.reset()
    agent.receive_state(init_state)
    model_free_train_loop(agent, world_model, False, Args().args.sim_trajectory_length)


def model_based_train_loop(agent: Agent, world_model: WorldModelInterface, env):
    storage = sample_env(agent, env)
    world_model.update(storage)

    for update in range(Args().args.model_updates):
        interact_world_model(agent, world_model)


