from src.enviroment import setup_env
from src.agent.agent import Agent
from src.agent.agentinterface import AgentInterface
from src.resultwriter import ResultWriter


def main():
    env = setup_env()
    agent = Agent()
    run_n_episodes(100, agent, env)
    env.close()


def run_n_episodes(episodes: int, agent: AgentInterface, env):
    results = ResultWriter("data", ["single"])                  # initialize data saving
    for episode in range(episodes):
        run_experiment(agent, env, results)
        #results.save_episode()                      # dynamically save gathered data


def run_experiment(agent: AgentInterface, env, results: ResultWriter):
    observation, info = env.reset()

    #agent.receive_state(observation)

    while True:
        #action = agent.select_action()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        print(observation.shape)
        #agent.receive_reward(reward)
        #agent.receive_state(observation)
        #agent.update_policy()

        #results.add_data(observation, action, reward)   # for the purpose of analysis.
        if terminated or truncated:
            return


if __name__ == '__main__':
    main()
