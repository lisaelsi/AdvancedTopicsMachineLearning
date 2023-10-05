import numpy as np
import random as rand


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q = np.random.uniform(0, 1, size=(state_space, action_space))
        self.Q[5] = [0] * action_space
        self.Q[7] = [0] * action_space
        self.Q[11] = [0] * action_space
        self.Q[12] = [0] * action_space
        self.Q[15] = [0] * action_space
        self.alpha = 0.1   # step size: (0, 1]
        self.epsilon = 0.05
        self.gamma = 0.95

        self.observation = None
        self.previous_observation = None

        self.action = None
        self.previous_action = None

    def observe(self, observation, reward, done):

        if self.previous_action is not None or self.previous_observation is not None:
            self.Q[self.previous_observation, self.previous_action] += self.alpha * (reward + self.gamma *
                                                                                     self.Q[self.observation, self.action] - self.Q[self.previous_observation, self.previous_action])

        if done:
            self.Q[self.previous_observation, self.previous_action] += self.alpha * \
                (reward - self.Q[self.previous_observation,
                 self.previous_action])


# S = previous observation
# A = previous action
# S' = observation
# A' = action


    def act(self, observation):

        self.previous_observation = self.observation

        # TODO - fixa detta?
        if isinstance(observation, tuple):
            self.observation = observation[0]
        else:
            self.observation = observation

        self.previous_action = self.action

        threshold = np.random.random()
        if threshold > self.epsilon:
            self.action = np.argmax(self.Q[self.observation])
        else:
            self.action = np.random.randint(self.action_space)

        return self.action
