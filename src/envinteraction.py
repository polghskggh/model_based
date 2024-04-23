import gymnasium as gym

from src.agent.agentinterface import AgentInterface
from src.pod.hyperparameters import hyperparameters
from src.resultwriter import ModelWriter
from src.worldmodel.worldmodelinterface import WorldModelInterface


def model_free_train_loop(agent: AgentInterface, env: gym.Env, results: ModelWriter):
    observation, _ = env.reset()
    agent.receive_state(observation)
    episode_return = 0

    for _ in range(hyperparameters["max_episode_length"]):
        reward, done = interact(agent, env, results)
        episode_return += reward
        if done:
            results.add_data(reward, "reward")
            results.add_data(episode_return, "return")
            return


def model_based_train_loop(agent: AgentInterface, world_model: WorldModelInterface, env: gym.Env, results: ModelWriter):
    sample_batches(agent, env, results)
    world_model.update()
    update_agent(agent, env, results)


def sample_batches(agent: AgentInterface, env: gym.Env, results: ModelWriter):
    episode_return = 0
    for _ in range(hyperparameters["max_episode_length"]):
        reward, done = interact(agent, env, results)
        episode_return += reward
        if done:
            results.add_data(reward, "reward")
            results.add_data(episode_return, "return")
            return episode_return


def update_agent(agent, env, results):
    for _ in range(hyperparameters["max_episode_length"]):
        interact(agent, env, update=True)


def interact(agent: AgentInterface, enviroment: WorldModelInterface|gym.Env, update: bool = True):
    action = agent.select_action()
    observation, reward, terminated, truncated, _ = enviroment.step(action)
    agent.receive_reward(reward)
    agent.receive_state(observation)

    if update:
        agent.update_policy()

    return reward, terminated or truncated