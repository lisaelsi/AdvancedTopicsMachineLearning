import numpy as np
import random as rand


class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q1 = np.random.uniform(1, 5, size=(state_space, action_space))
        self.Q2 = np.random.uniform(1, 5, size=(state_space, action_space))

        self.Q1[5] = [0] * action_space
        self.Q1[7] = [0] * action_space
        self.Q1[11] = [0] * action_space
        self.Q1[12] = [0] * action_space
        self.Q1[15] = [0] * action_space
        self.Q2[5] = [0] * action_space
        self.Q2[7] = [0] * action_space
        self.Q2[11] = [0] * action_space
        self.Q2[12] = [0] * action_space
        self.Q2[15] = [0] * action_space
        self.alpha = 0.1    # step size: (0, 1]
        self.epsilon = 0.05
        self.gamma = 0.95
        self.previous_observation = None
        self.previous_action = None

    def observe(self, observation, reward, done):

        flip = rand.randint(0, 2)
        if flip == 0:
            Q_update = self.Q1
            Q_other = self.Q2
        else:
            Q_update = self.Q2
            Q_other = self.Q1

        if done:
            Q_update[self.previous_observation, self.previous_action] += self.alpha * \
                (reward - Q_update[self.previous_observation,
                 self.previous_action])
        else:
            Q_update[self.previous_observation, self.previous_action] += self.alpha * (reward + self.gamma * np.max(
                Q_update[observation, np.argmax(Q_other[observation])]) - Q_update[self.previous_observation, self.previous_action])

    def act(self, observation):

        if isinstance(observation, tuple):
            self.previous_observation = observation[0]
        else:
            self.previous_observation = observation

        threshold = np.random.random()
        if threshold > self.epsilon:
            Q = np.add(self.Q1, self.Q2)
            self.previous_action = np.argmax(Q[self.previous_observation])
        else:
            self.previous_action = np.random.randint(self.action_space)

        return self.previous_action
