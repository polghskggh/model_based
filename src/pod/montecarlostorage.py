import jax.numpy as jnp

from src.enviroment import Shape
from src.singletons.hyperparameters import Args


class MonteCarloStorage:
    def __init__(self):
        self.old_log_odds = None
        self.states, self.actions, self.rewards, self.dones = None, None, None, None
        self.reset()

    # add new data
    def add_transition(self, state, action, reward, done, old_log_odds):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.old_log_odds.append(old_log_odds)

    def pad(self):
        while len(self.rewards) < Args().args.trajectory_length:
            self.rewards.append(jnp.zeros(self.rewards[-1].shape))
            self.actions.append(jnp.zeros(self.actions[-1].shape))
            self.states.append(self.states[-1])
            self.dones.append(jnp.ones(self.dones[-1].shape))
            self.old_log_odds.append(self.old_log_odds[-1])

    def end_episode(self, final_state):
        self.states.append(final_state)
        self.pad()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_odds = []

    def data(self):
        return (jnp.array(self.states), jnp.array(self.actions),
                jnp.array(self.rewards), jnp.array(self.dones),
                jnp.array(self.old_log_odds))
