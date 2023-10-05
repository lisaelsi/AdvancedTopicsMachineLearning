import numpy as np
import random as rand


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q = np.random.uniform(50, 100, size=(state_space, action_space))
        self.Q[5] = [0] * action_space
        self.Q[7] = [0] * action_space
        self.Q[11] = [0] * action_space
        self.Q[12] = [0] * action_space
        self.Q[15] = [0] * action_space
        self.alpha = 0.1   # step size: (0, 1]
        self.epsilon = 0.05
        self.gamma = 0.95
        self.previous_observation = None
        self.previous_action = None

    def observe(self, observation, reward, done):

        if done:
            self.Q[self.previous_observation, self.previous_action] += self.alpha * \
                (reward - self.Q[self.previous_observation,
                 self.previous_action])

        else:
            self.Q[self.previous_observation, self.previous_action] += self.alpha * (reward + self.gamma * np.max(
                self.Q[observation]) - self.Q[self.previous_observation, self.previous_action])

    def act(self, observation):

        if isinstance(observation, tuple):
            self.previous_observation = observation[0]
        else:
            self.previous_observation = observation

        threshold = np.random.random()
        if threshold > self.epsilon:
            self.previous_action = np.argmax(self.Q[self.previous_observation])
        else:
            self.previous_action = np.random.randint(self.action_space)

        return self.previous_action
