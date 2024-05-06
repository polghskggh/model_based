import jax.numpy as jnp

from src.enviroment import Shape
from src.pod.hyperparameters import hyperparameters


class MonteCarloStorage:
    def __init__(self):
        self.trajectory_length = hyperparameters["max_episode_length"]
        self.num_of_trajectories = 0
        self.states = [[]]
        self.actions = [[]]
        self.rewards = [[]]

    # add new data
    def add_transition(self, state, action, reward):
        self.states[self.num_of_trajectories].append(state)
        self.actions[self.num_of_trajectories].append(action)
        self.rewards[self.num_of_trajectories].append(reward)

    def pad(self):
        while len(self.rewards[self.num_of_trajectories]) < self.trajectory_length:
            self.rewards[self.num_of_trajectories].append(0)
            self.actions[self.num_of_trajectories].append(0)
            self.states[self.num_of_trajectories].append(self.states[self.num_of_trajectories][-1])

    def end_episode(self, final_state):
        self.states[self.num_of_trajectories].append(final_state)
        self.pad()

        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.num_of_trajectories += 1

    def reset(self):
        self.rewards = [[]]
        self.states = [[]]
        self.actions = [[]]
        self.num_of_trajectories = 0

    def __getitem__(self, item):
        return self.states[item], self.actions[item], self.rewards[item]

    def data(self):
        return jnp.array(self.states[:-1]), jnp.array(self.actions[:-1]), jnp.array(self.rewards[:-1])
