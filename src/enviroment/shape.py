class Shape(tuple):
    shape: tuple = None

    def __new__(cls, env=None):
        if env is not None:
            cls.shape = (env.observation_space.shape, env.action_space.n)
        return cls.shape
