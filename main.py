from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.enviroment import make_env
from src.gpu import setup_gpu, check_gpu
from src.resultwriter.modelwriter import writer_instances, ModelWriter
from gymnasium import Env


def main():
    env = make_env()
    agent = Agent("dqn")
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
    agent.receive_state(observation)

    for _ in range(5000):
        action = agent.select_action()
        observation, reward, terminated, truncated, _ = env.step(action)
        agent.receive_reward(reward)
        agent.receive_state(observation)
        agent.update_policy()
        results.add_data(reward)   # for the purpose of analysis.
        if terminated or truncated:
            return


if __name__ == '__main__':
    check_gpu()
    main()
