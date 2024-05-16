import jax.numpy as jnp

from src.enviroment import Shape
from src.singletons.hyperparameters import Args


class MonteCarloStorage:
    def __init__(self, batch_size: int = 1):
        self.old_log_odds = None
        self.batch_size = batch_size
        self.num_of_trajectories = 0
        self.states, self.actions, self.rewards, self.dones = None, None, None, None
        self.reset()

    # add new data
    def add_transition(self, state, action, reward, done, old_log_odds):
        if self.batch_size == 1:
            self.states[self.num_of_trajectories].append(state)
            self.actions[self.num_of_trajectories].append(action)
            self.rewards[self.num_of_trajectories].append(reward)
            self.dones[self.num_of_trajectories].append(done)
            self.old_log_odds[self.num_of_trajectories].append(old_log_odds)
        else:
            for i in range(self.batch_size):
                self.states[self.num_of_trajectories + i].append(state[i])
                self.actions[self.num_of_trajectories + i].append(action[i])
                self.rewards[self.num_of_trajectories + i].append(reward[i])
                self.dones[self.num_of_trajectories + i].append(done[i])
                self.old_log_odds[self.num_of_trajectories].append(old_log_odds[i])

    def pad(self):
        while len(self.rewards[self.num_of_trajectories]) < Args().args.max_trajectory_length:
            for i in range(self.batch_size):
                self.rewards[self.num_of_trajectories + i].append(0)
                self.actions[self.num_of_trajectories + i].append(0)
                self.states[self.num_of_trajectories + i].append(self.states[self.num_of_trajectories + i][-1])
                self.dones[self.num_of_trajectories + i].append(True)
                self.old_log_odds[self.num_of_trajectories + i].append(self.old_log_odds[self.num_of_trajectories + i][-1])

    def end_episode(self, final_state):
        if self.batch_size == 1:
            self.states[self.num_of_trajectories].append(final_state)
        else:
            for i in range(self.batch_size):
                self.states[self.num_of_trajectories + i].append(final_state[i])

        self.pad()
        self.states.extend([] for _ in range(self.batch_size))
        self.actions.extend([] for _ in range(self.batch_size))
        self.rewards.extend([] for _ in range(self.batch_size))
        self.dones.extend([] for _ in range(self.batch_size))
        self.old_log_odds.extend([] for _ in range(self.batch_size))
        self.num_of_trajectories += self.batch_size

    def reset(self):
        self.num_of_trajectories = 0
        self.states = [[] for _ in range(self.batch_size)]
        self.actions = [[] for _ in range(self.batch_size)]
        self.rewards = [[] for _ in range(self.batch_size)]
        self.dones = [[] for _ in range(self.batch_size)]
        self.old_log_odds = [[] for _ in range(self.batch_size)]

    def data(self):
        return (jnp.array(self.states[:-self.batch_size]), jnp.array(self.actions[:-self.batch_size]),
                jnp.array(self.rewards[:-self.batch_size]), jnp.array(self.dones[:-self.batch_size]),
                jnp.array(self.old_log_odds[:-self.batch_size]))
