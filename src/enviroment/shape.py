class Shape(tuple):
    shape: tuple = None

    @staticmethod
    def initialize(env):
        Shape.shape = (env.observation_space.shape, env.action_space.n)

    def __new__(cls):
        return cls.shape
