from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.enviroment import make_env, make_data_env
from src.gpu import setup_gpu, check_gpu
from src.pod.trajectorystorage import TrajectoryStorage
from src.resultwriter.modelwriter import writer_instances, ModelWriter
from gymnasium import Env


data_gatherer = TrajectoryStorage("trajectory")


def main():
    env = make_data_env()
    agent = Agent("atari-ddpg")
    run_n_episodes(100, agent, env)
    env.close()


def run_n_episodes(episodes: int, agent: AgentInterface, env: Env):
    results = writer_instances["reward"]
    for episode in range(episodes):
        run_experiment(agent, env, results)
        results.save_all()                      # dynamically save gathered data
        results.flush_all()


def run_experiment(agent: AgentInterface, env: Env, results: ModelWriter):
    observation, _ = env.reset()
    observation, last_target = observation
    agent.receive_state(observation)

    for _ in range(1000):
        action = agent.select_action()
        data_gatherer.add_input(observation, action)

        observation, reward, terminated, truncated, _ = env.step(action)
        observation, last_target = observation

        data_gatherer.add_teacher(reward, last_target)
        data_gatherer.flush_buffer()

        agent.receive_reward(reward)
        agent.receive_state(observation)
        agent.update_policy()
        results.add_data(reward)   # for the purpose of analysis.
        if terminated or truncated:
            return


if __name__ == '__main__':
    check_gpu()
    main()
