import torch
import torch.optim as optim
import copy
from agents.ddpg import Actor, Critic, Policy
from environment.config import ALERTS
from numpy.typing import NDArray
from environment.config import DEFENDER_BUDGET_DEFAULT
import numpy as np


def preprocess_defender_state(state):
    return np.log1p(state).astype(np.float32) / 10.0


def enforce_defender_budget(raw_scores, budget, alerts_available):
    raw_scores = np.maximum(raw_scores, 0)

    if raw_scores.sum() == 0:
        raw_scores = np.ones_like(raw_scores)

    frac = raw_scores / raw_scores.sum()

    alloc = np.round(frac * budget).astype(np.int32)

    alloc = np.minimum(alloc, alerts_available)

    return alloc


def run_defender_policy(policy: Policy, state):
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

        return enforce_defender_budget(out, DEFENDER_BUDGET_DEFAULT, state)

    else:
        raise ValueError("Unknown policy type")


class Defender:
    def __init__(self):
        self.policy = Actor(len(ALERTS), len(ALERTS))
        self.value = Critic(len(ALERTS), len(ALERTS))
        self.policy_tool = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_tool = optim.Adam(self.value.parameters(), lr=2e-3)
        self.gamma = 0.95
        self.policies = []
        self.action_dim = len(ALERTS)

    def update(self, states, actions, rewards, next_states):
        """
        states      : (B, state_dim)
        actions     : (B, action_dim)
        rewards     : (B,)
        next_states : (B, state_dim)
        """
        # -----------------------------------
        # convert batch arrays to tensors
        # -----------------------------------
        s = torch.tensor(states, dtype=torch.float32)

        a = torch.tensor(actions, dtype=torch.float32)

        r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

        s2 = torch.tensor(next_states, dtype=torch.float32)

        # -----------------------------------
        # Critic update
        # -----------------------------------
        next_action = self.policy(s2)

        target = r + self.gamma * self.value(s2, next_action).detach()

        estimate = self.value(s, a)

        loss_value = ((estimate - target) ** 2).mean()

        self.value_tool.zero_grad()
        loss_value.backward()
        self.value_tool.step()

        # -----------------------------------
        # Actor update
        # -----------------------------------
        chosen_action = self.policy(s)

        loss_policy = -self.value(s, chosen_action).mean()

        self.policy_tool.zero_grad()
        loss_policy.backward()
        self.policy_tool.step()

    def save_policy(self, itr):
        snapshot = copy.deepcopy(self.policy.state_dict())

        self.policies.append(Policy(model=snapshot, itr=itr, type="nn"))


def uniform_policy(n: NDArray):
    inv = np.zeros_like(n, dtype=np.int32)
    inv = inv + DEFENDER_BUDGET_DEFAULT // len(n)
    inv = np.clip(inv, np.zeros_like(inv), n)
    inv = inv.astype(np.int32)
    return inv


def priority_policy(n):
    """
    Put budget on largest backlog first
    """

    inv = np.zeros_like(n, dtype=np.int32)

    budget = DEFENDER_BUDGET_DEFAULT

    order = np.argsort(-n)  # descending

    for idx in order:
        take = min(n[idx], budget)
        inv[idx] = take
        budget -= take

        if budget == 0:
            break

    return inv
