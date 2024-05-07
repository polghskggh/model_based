from src.agent.critic import CriticInterface


class DreamerCritic(CriticInterface):
    def provide_feedback(self, *args):
        pass

    def calculate_grads(self, *args):
        pass

    def update(self, grads: dict):
        pass

    def save(self):
        pass

    def load(self):
        pass