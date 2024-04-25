import jax.numpy as jnp


class TrajectoryStorage:
    def __init__(self):
        self.frame_stack = []
        self.actions = []
        self.rewards = []
        self.next_frames = []
        self.size = 0
        self.index = 0
        # flush data into file

    def store(self):
        data = {"frame_stack": self.frame_stack, "action": self.actions,
                "reward": self.rewards, "next_frame": self.next_frames}
        jnp.savez(f"data_{self.index}", **data)
        self.index += 1
        self.rewards = []
        self.frame_stack = []
        self.actions = []
        self.next_frames = []

    # add new data
    def add_input(self, frame_stack, action):
        self.frame_stack.append(frame_stack)
        self.actions.append(action)
        self.size += 1

    def add_teacher(self, reward, next_frame):
        self.rewards.append(reward)
        self.next_frames.append(next_frame)
        if self.size == 500:
            self.store()
            self.size = 0