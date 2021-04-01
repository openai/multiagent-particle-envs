from multiagent.policy.policy import Policy
import numpy as np
from collections import defaultdict
import random


class EpsilonGreedyPolicy(Policy):
    def __init__(self, env, agent_index):
        super(EpsilonGreedyPolicy, self).__init__()
        self.env = env

    def action(self, obs, q_vals, epsilon):
        if random.random() < epsilon:
            action = np.random.choice(len(q_vals[obs]))  # randomly select action from state
        else:
            action = np.argmax(q_vals[obs])  # greedily select action from state
        return action
