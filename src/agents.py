import numpy as np

class Agent(object):
    def __init__(self, observation_space: int, action_space: int):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state: np.ndarray) -> np.ndarray:
        pass


class RandomAgent(Agent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)

    def act(self, state: np.ndarray) -> np.ndarray:
        return