import numpy as np
from gym import Space
from gymnasium import ObservationWrapper


class ObservationReshape(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Space(shape=env.observation_space.shape[1:3] +
                                       (env.observation_space.shape[0] * env.observation_space.shape[3], ),
                                       dtype=env.observation_space.dtype)

    def observation(self, observation: "LazyFrames"):
        """
        transform observation from (Stack, Height, Width, Channel)
                                to (Height, Width, Channel * Stack)

        :param observation: observation from the environment
        :return: reshaped observation
        """
        new_shape = observation.shape[1:3] + (observation.shape[0] * observation.shape[3],)
        return np.array(observation).transpose(1, 2, 0, 3).reshape(new_shape)
