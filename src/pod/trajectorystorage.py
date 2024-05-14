import random

import jax
import jax.numpy as jnp
import jax.random as jr
from src.enviroment import Shape
from src.pod.hyperparameters import hyperparameters


class TrajectoryStorage:
    def __init__(self, batch_size: int):
        self.frame_stack = []
        self.actions = []
        self.rewards = []
        self.next_frames = []
        self.size = 0
        self.batch_size = batch_size
        self.index = 0
        self._key = jr.PRNGKey(0)
        # flush data into file

    # add new data
    # TODO: input shape (2, 105, 80, 12) (2, ) (2, 1) (2, 105, 80, 12), deal with batch transitions
    def add_transition(self, frame_stack, action, reward, next_frame):
        self.frame_stack.append(frame_stack)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_frames.append(next_frame)
        self.size += 1

    def reset(self):
        self.frame_stack = []
        self.actions = []
        self.rewards = []
        self.next_frames = []
        self.size = 0

    def store(self):
        data = {"frame_stack": self.frame_stack, "action": self.actions,
                "reward": self.rewards, "next_frame": self.next_frames}
        jnp.savez(f"data_{self.index}", **data)
        self.index += 1
        self.reset()

    def batched_data(self):
        for index in range(0, self.size, self.batch_size):
            end_index = min(index + self.batch_size, self.size)
            yield (jnp.array(self.frame_stack[index: end_index], dtype=jnp.float32),
                   jnp.array(self.actions[index: end_index], dtype=jnp.float32),
                   jnp.array(self.rewards[index: end_index], dtype=jnp.float32),
                   jnp.array(self.next_frames[index: end_index], dtype=jnp.float32))

    def data(self):
        return self.frame_stack, self.actions, self.rewards, self.next_frames

    def sample_states(self, n: int) -> list[jax.Array]:
        return random.sample(self.next_frames, n)
