import gymnasium as gym

from src.agent.agentinterface import AgentInterface
from src.modelfree import interact
from src.pod.hyperparameters import hyperparameters
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances
from src.worldmodel.worldmodelinterface import WorldModelInterface


def model_based_train_loop(agent: AgentInterface, world_model: WorldModelInterface, env: gym.Env):
    data = sample_batches(agent, env)
    world_model.update(data)
    update_agent(agent, world_model)


def sample_batches(agent: AgentInterface, env: gym.Env):
    episode_return = 0
    for _ in range(hyperparameters["world"]["batch"]):
        reward, done = interact(agent, env)
        episode_return += reward
        writer_instances["reward"].add_data(reward, "reward")
        if done:
            writer_instances["reward"].add_data(episode_return, "return")
            episode_return = 0
            env.reset()

    return agent.replay_buffer


def update_agent(agent: AgentInterface, env: WorldModelInterface):
    for _ in range(hyperparameters["max_episode_length"]):
        interact(agent, env, update=True)
