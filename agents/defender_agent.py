import torch
import torch.optim as optim
import copy
from ddpg import Actor,Critic,Policy
from environment.config import ALERTS

class Defender:
    def __init__(self):
        self.policy = Actor(len(ALERTS),len(ALERTS))
        self.value = Critic(len(ALERTS),len(ALERTS))
        self.policy_tool = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_tool = optim.Adam(self.value.parameters(), lr=2e-3)
        self.gamma = 0.95
        self.policies = []

    def update(self,state,action,reward,next_state):
        """
        Implements a single update step of the DDPG mix algorithm
        """
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        r = torch.tensor([[reward]], dtype=torch.float32)
        s2 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # next action from current policy
        next_action = self.policy(s2)

        # target value = r + gamma * Q(next_state,next_action)
        target = r + self.gamma * self.value(s2, next_action).detach()

        # current estimate
        estimate = self.value(s, a)

        # squared error
        loss_value = ((estimate - target) ** 2).mean()

        self.value_tool.zero_grad()
        loss_value.backward()
        self.value_tool.step()

    def save_policy(self, itr):
        snapshot = copy.deepcopy(self.policy.state_dict())

        self.policies.append(
            Policy(model=snapshot, itr=itr)
        )