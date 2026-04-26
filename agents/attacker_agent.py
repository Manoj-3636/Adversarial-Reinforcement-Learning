import numpy as np
import torch
import torch.optim as optim
import copy

from agents.ddpg import Actor, Critic, Policy
from environment.config import (
    ALERTS,
    ATTACKS,
    ATTACKER_BUDGET_DEFAULT,
    ATTACK_LOSSES,
    ATTACK_COSTS,
)


def preprocess_attacker_state(N, M, S):
    x = np.concatenate(
        [np.log1p(N) / 10.0, M.astype(np.float32), np.log1p(S.flatten()) / 10.0]
    )
    return x.astype(np.float32)


def enforce_attacker_budget(
    raw_scores,
    costs=np.array(list(ATTACK_COSTS.values())),
    budget=ATTACKER_BUDGET_DEFAULT,
):
    """
    raw_scores : preference scores from actor
    costs      : integer cost per attack
    budget     : total attacker resource budget

    returns bool vector
    """

    raw_scores = np.maximum(np.asarray(raw_scores, dtype=float), 0.0)
    costs = np.asarray(costs)
    ratio = raw_scores / (costs + 1e-8)
    order = np.argsort(-ratio)

    action = np.zeros(len(raw_scores), dtype=bool)

    remain = budget

    for idx in order:

        if costs[idx] <= remain:
            action[idx] = True
            remain -= costs[idx]

    return action


def run_attacker_policy(policy: Policy, N, M, S):
    """
    full attacker observation
    returns attack action vector
    """

    if policy.type == "func":
        return policy.model(N, M, S)

    elif policy.type == "nn":

        x = preprocess_attacker_state(N, M, S)

        state_dim = len(ALERTS) + len(ATTACKS) + len(ALERTS) * len(ATTACKS)

        net = Actor(state_dim, len(ATTACKS))
        net.load_state_dict(policy.model)
        net.eval()

        with torch.no_grad():
            inp = torch.tensor(x).unsqueeze(0)
            out = net(inp).squeeze(0).numpy()

        return enforce_attacker_budget(out)

    else:
        raise ValueError("Unknown policy type")


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

        self.state_dim = len(ALERTS) + len(ATTACKS) + len(ATTACKS) * len(ALERTS)

        self.action_dim = len(ATTACKS)

        # attack policy π(s)
        self.policy = Actor(self.state_dim, self.action_dim)

        # attack value Q(s,a)
        self.value = Critic(self.state_dim, self.action_dim)

        self.policy_tool = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.value_tool = optim.Adam(self.value.parameters(), lr=2e-3)

        self.gamma = 0.95

        self.policies = []

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
        # target = r + gamma * Q(s',pi(s'))
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
        # maximize Q(s, pi(s))
        # -----------------------------------
        chosen_action = self.policy(s)

        loss_policy = -self.value(s, chosen_action).mean()

        self.policy_tool.zero_grad()
        loss_policy.backward()
        self.policy_tool.step()

    def save_policy(self, itr):
        snapshot = copy.deepcopy(self.policy.state_dict())

        self.policies.append(Policy(model=snapshot, itr=itr, type="nn"))


def greedy_attacker(N, M, S):
    return np.array([False, True, False, False, True, False, False])


def uniform_attacker(N, M, S):
    x = np.zeros(len(ATTACKS), dtype=bool)
    x[0] = True
    return x
