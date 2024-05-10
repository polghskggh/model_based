import gymnasium as gym
from jax import lax

from src.agent.agent import Agent
from src.modelfree import interact
from src.pod.hyperparameters import hyperparameters
from src.pod.trajectorystorage import TrajectoryStorage
from src.resultwriter.modelwriter import writer_instances
from src.worldmodel.worldmodelinterface import WorldModelInterface


def model_based_train_loop(agent: Agent, world_model: WorldModelInterface, env: gym.Env):
    data = sample_batches(agent, env)
    world_model.update(data)
    print("Model updated")
    update_agent(agent, world_model)


def sample_batches(agent: Agent, env: gym.Env):
    data_storage = TrajectoryStorage(hyperparameters["simple"]["batch_size"])
    episode_return = 0
    observation, _ = env.reset()
    agent.receive_state(observation)

    for _ in range(hyperparameters["simple"]["data_size"]):
        _, done = interact(agent, env, False)

        stack, action, reward, next_stack = agent.last_transition()
        next_frame = lax.dynamic_slice_in_dim(next_stack, -3, 3, axis=-1)
        data_storage.add_transition(stack, action, reward, next_frame)

        episode_return += reward
        writer_instances["reward"].add_data(reward, "reward")

        if done:
            writer_instances["reward"].add_data(episode_return, "return")
            episode_return = 0
            observation, _ = env.reset()
            agent.receive_state(observation)

    return data_storage


def update_agent(agent: Agent, env: WorldModelInterface):
    for time_step in range(hyperparameters["max_episode_length"]):
        interact(agent, env, update=True)
