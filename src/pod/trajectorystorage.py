import jax.numpy as jnp

from src.enviroment import Shape
from src.pod.hyperparameters import hyperparameters


class TrajectoryStorage:
    def __init__(self):
        self.frame_stack = []
        self.actions = []
        self.rewards = []
        self.next_frames = []
        self.size = 0
        self.index = 0
        # flush data into file

    # add new data
    def add_transition(self, frame_stack, action, reward, next_frame):
        self.frame_stack.append(frame_stack)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_frames.append(next_frame)
        self.size += 1

    def reset(self):
        del self.rewards[:]
        del self.frame_stack[:]
        del self.actions[:]
        del self.next_frames[:]
        self.size = 0

    def store(self):
        data = {"frame_stack": self.frame_stack, "action": self.actions,
                "reward": self.rewards, "next_frame": self.next_frames}
        jnp.savez(f"data_{self.index}", **data)
        self.index += 1
        self.reset()

    def data(self):
        return self.frame_stack, self.actions, self.rewards, self.next_frames
