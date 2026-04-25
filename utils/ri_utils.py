import random
from collections import deque

import numpy as np


class ReplayBuffer :
    """
    Experience Replay Buffer used by both the Attacker and Defender
    DDPG-MIX agents to store and sample state transitions.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state,action,reward, next_state

    def __len__(self):
        return len(self.buffer)

class EpsilonGreedyExploration :
    """
    Experience Replay Buffer used by both the Attacker and Defender
    DDPG-MIX agents to store and sample state transitions.
    """

    def init(self,epsilon_start = 1.0 ,epsilon_min = 0.01 ,decay_rate=0.995):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

    def select_action(self, policy_action, action_space_dim):
        if(random.random() < self.epsilon):
            random_action = np.random.rand(action_space_dim)
            return random_action / np.sum(random_action)

        return policy_action

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)





