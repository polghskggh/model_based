import jax

from src.enviroment import Enviroment
from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.resultwriter.modelwriter import writer_instances, ModelWriter
from jax import devices


def main():
    check_gpu()
    env = Enviroment()
    agent = Agent("atari-ddpg")
    run_n_episodes(100, agent, env)
    env.close()


def run_n_episodes(episodes: int, agent: AgentInterface, env: Enviroment):
    results = writer_instances["reward"]
    for episode in range(episodes):
        run_experiment(agent, env, results)
        results.save_episode()                      # dynamically save gathered data
        results.flush_all()


def run_experiment(agent: AgentInterface, env: Enviroment, results: ModelWriter):
    observation, _ = env.reset()
    agent.receive_state(observation)

    for _ in range(1000):
        action = agent.select_action()
        observation, reward, terminated, truncated, _ = env.step(action)
        agent.receive_reward(1)
        agent.receive_state(observation)
        agent.update_policy()
        results.add_data(1)   # for the purpose of analysis.
        results.save_episode()
        if terminated or truncated:
            return


def check_gpu():
    try:
        gpu_devices = devices('gpu')
        print("GPU is available.")
        for gpu in gpu_devices:
            print(gpu)
    except RuntimeError:
        print("No GPU available.")


if __name__ == '__main__':
    main()
