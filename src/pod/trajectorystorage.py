from src.resultwriter.csvwriter import CsvWriter


class TrajectoryStorage:
    def __init__(self, filename):
        headers = ["frame_stack", "action", "reward", "next_frame"]
        self.csv_writer = CsvWriter(filename, headers)
        self.frame_stack = []
        self.actions = []
        self.rewards = []
        self.next_frames = []
        self.size = 0
        # flush data into file

    def flush_buffer(self):
        data = self._prepare_data()
        self.csv_writer.store_data(data)
        self.rewards = []
        self.frame_stack = []
        self.actions = []
        self.next_frames = []

    def _prepare_data(self):
        data = [[stack, action, reward, nextFrame]
                for stack, action, reward, nextFrame
                in zip(self.frame_stack, self.actions, self.rewards, self.next_frames)]
        return data

    # add new data
    def add_input(self, frame_stack, action):
        self.frame_stack.append(frame_stack)
        self.actions.append(action)
        self.size += 1

    def add_teacher(self, reward, next_frame):
        self.rewards.append(reward)
        self.next_frames.append(next_frame)
        if self.size == 50:
            self.flush_buffer()
            self.size = 0