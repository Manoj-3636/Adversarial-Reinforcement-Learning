import torch
import torch.optim as optim
import copy
from agents.ddpg import Actor,Critic,Policy
from environment.config import ALERTS
from numpy.typing import NDArray
from environment.config import DEFENDER_BUDGET_DEFAULT
import numpy as np

def preprocess_defender_state(state):
    return np.log1p(state).astype(np.float32) / 10.0

def run_defender_policy(policy:Policy, state):
    """
    state = uninvestigated_alerts
    returns defense action
    """

    if policy.type == "func":
        return policy.model(state)

    elif policy.type == "nn":

        x = preprocess_defender_state(state)

        net = Actor(len(ALERTS), len(ALERTS))
        net.load_state_dict(policy.model)
        net.eval()

        with torch.no_grad():
            inp = torch.tensor(x).unsqueeze(0)
            out = net(inp).squeeze(0).numpy()

        return out

    else:
        raise ValueError("Unknown policy type")

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
        Implements a single update step of the DDPG mix algorithm only give preprocessed states
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
            Policy(model=snapshot, itr=itr,type="nn")
        )


def uniform_policy(n:NDArray):
    inv = np.zeros_like(n,dtype=np.int32)
    inv = inv + DEFENDER_BUDGET_DEFAULT//len(n)
    inv = np.clip(inv,np.zeros_like(inv),n)
    inv = inv.astype(np.int32)
    return inv

def priority_policy(n):
    """
    Put budget on largest backlog first
    """

    inv = np.zeros_like(n, dtype=np.int32)

    budget = DEFENDER_BUDGET_DEFAULT

    order = np.argsort(-n)   # descending

    for idx in order:
        take = min(n[idx], budget)
        inv[idx] = take
        budget -= take

        if budget == 0:
            break

    return inv