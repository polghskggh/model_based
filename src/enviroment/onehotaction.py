from gymnasium import ActionWrapper

import numpy as np


class OneHotAction(ActionWrapper):
    def action(self, action):
        """
        Convert one hot action to integer action

        :param action: one hot action
        :return: integer action
        """
        action = np.argmax(action)
        return action
