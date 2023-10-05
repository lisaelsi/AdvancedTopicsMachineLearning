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
        self.previous_observation = None
        self.previous_action = None

    def observe(self, observation, reward, done):

        state_probabilites = self.get_state_probabilites(
            self.epsilon, observation)

        q_values = [self.Q[observation, action]
                    for action in range(self.action_space)]
        state_expectation = sum(
            [prob * val for prob, val in zip(state_probabilites, q_values)])

        if done:
            self.Q[self.previous_observation, self.previous_action] += self.alpha * \
                (reward - self.Q[self.previous_observation,
                 self.previous_action])
        else:
            self.Q[self.previous_observation, self.previous_action] += self.alpha * \
                (reward + self.gamma * state_expectation -
                 self.Q[self.previous_observation, self.previous_action])

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

    def get_state_probabilites(self, epsilon, observation):
        probs = [epsilon/self.action_space] * self.action_space
        best_action = np.argmax(self.Q[observation])
        probs[best_action] += (1.0 - epsilon)

        return probs
