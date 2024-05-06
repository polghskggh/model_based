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

    def __getitem__(self, item):
        return (jnp.array(self.frame_stack[item]), jnp.array(self.actions[item]),
                jnp.array(self.rewards[item]), jnp.array(self.next_frames[item]))

    def episodic_data(self):
        trajectory_length = hyperparameters["ppo"]["trajectory_length"]
        episodes = self.size // trajectory_length
        return (jnp.array(self.frame_stack).reshape(episodes, trajectory_length, *Shape()[0]),
                jnp.array(self.actions).reshape(episodes, trajectory_length, 1),
                jnp.array(self.rewards).reshape(episodes, trajectory_length, 1),
                jnp.array(self.next_frames).reshape(episodes, trajectory_length, *Shape()[0]))
