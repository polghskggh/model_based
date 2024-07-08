from singleton_decorator import singleton


@singleton
class StepTracker:
    value: int = 0

    def increment(self, value: int = 1):
        self.value += value

    def __int__(self):
        return self.value


@singleton
class ModelEnvTracker:
    value: int = 0

    def increment(self, value: int = 1):
        self.value += value

    def __int__(self):
        return self.value
