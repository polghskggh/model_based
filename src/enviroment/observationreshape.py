from gymnasium import ObservationWrapper
import numpy as np


class ObservationReshape(ObservationWrapper):
    def observation(self, observation: "LazyFrames"):
        """
        transform observation from (Stack, Height, Width, Channel)
                                to (Height, Width, Channel * Stack)

        :param observation: observation from the environment
        :return: reshaped observation
        """
        new_shape = observation.shape[1:3] + (observation.shape[0] * observation.shape[3],)
        return np.array(observation).transpose(1, 2, 0, 3).reshape(new_shape)


class ObservationTrajectories(ObservationWrapper):
    def observation(self, observation: "LazyFrames") -> tuple:
        """
        transform observation from (Stack, Height, Width, Channel)
                                to (Height, Width, Channel * Stack)

        :param observation: observation from the environment
        :return: reshaped observation
        """
        current_frame = observation[3]
        new_shape = observation.shape[1:3] + (observation.shape[0] * observation.shape[3],)
        return np.array(observation).transpose(1, 2, 0, 3).reshape(new_shape), current_frame
