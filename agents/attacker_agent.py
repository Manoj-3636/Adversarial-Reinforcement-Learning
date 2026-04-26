import numpy as np
import torch
import torch.optim as optim
import copy

from ddpg import Actor, Critic, Policy
from environment.config import ALERTS, ATTACKS,ATTACKER_BUDGET_DEFAULT,ATTACK_LOSSES

class Attacker:
    def __init__(self):
        """
        According to paper:

        Attacker observes full state:
            N = uninvestigated alerts              -> |T|
            M = attacks mounted previous round     -> |A|
            S = alerts caused by attacks matrix    -> |A| x |T|

        Total state dimension:

            |T| + |A| + |A||T|
        """

        self.state_dim = (
                len(ALERTS)
                + len(ATTACKS)
                + len(ATTACKS) * len(ALERTS)
        )

        self.action_dim = len(ATTACKS)

        # attack policy π(s)
        self.policy = Actor(self.state_dim, self.action_dim)

        # attack value Q(s,a)
        self.value = Critic(self.state_dim, self.action_dim)

        self.policy_tool = optim.Adam(
            self.policy.parameters(),
            lr=1e-3
        )

        self.value_tool = optim.Adam(
            self.value.parameters(),
            lr=2e-3
        )

        self.gamma = 0.95

        self.policies = []

    def update(self, state, action, reward, next_state):
        """
        One DDPG-MIX update step for attacker

        state      = flattened full attacker state
        action     = attack allocation vector
        reward     = attacker reward
        next_state = next flattened state
        """

        s = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        a = torch.tensor(
            action,
            dtype=torch.float32
        ).unsqueeze(0)

        r = torch.tensor(
            [[reward]],
            dtype=torch.float32
        )

        s2 = torch.tensor(
            next_state,
            dtype=torch.float32
        ).unsqueeze(0)

        # next action from current policy
        next_action = self.policy(s2)

        # TD target
        target = r + self.gamma * self.value(
            s2, next_action
        ).detach()

        # current estimate
        estimate = self.value(s, a)

        # fit critic
        loss_value = ((estimate - target) ** 2).mean()

        self.value_tool.zero_grad()
        loss_value.backward()
        self.value_tool.step()

        # improve policy
        chosen_action = self.policy(s)

        loss_policy = -self.value(
            s, chosen_action
        ).mean()

        self.policy_tool.zero_grad()
        loss_policy.backward()
        self.policy_tool.step()

    def save_policy(self, itr):
        snapshot = copy.deepcopy(
            self.policy.state_dict()
        )

        self.policies.append(
            Policy(model=snapshot, itr=itr,type="nn")
        )

def greedy_attacker():
    return np.array([False,True,False,False,True,False,False])