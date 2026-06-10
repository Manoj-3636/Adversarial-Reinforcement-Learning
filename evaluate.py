import torch

from agents.attacker_agent import preprocess_attacker_state
from environment.config import ATTACK_COSTS, ATTACKER_BUDGET_DEFAULT
from environment.ade import ADE
import numpy as np
from agents.defender_agent import Defender, run_defender_policy, uniform_policy
from agents.attacker_agent import greedy_attacker


def sample_policy(pool, sigma):
    idx = np.random.choice(len(pool), p=sigma)

    return pool[idx]


def uniform_attacker_policy(N, M, S):
    costs = np.array(list(ATTACK_COSTS.values()))
    action = np.zeros(len(costs), dtype=bool)
    budget = ATTACKER_BUDGET_DEFAULT
    order = np.random.permutation(len(costs))
    for idx in order:
        if costs[idx] <= budget:
            action[idx] = True
            budget -= costs[idx]
    return action


def greedy_attacker_vs_arl_defender(trainer, sigma_D, episodes: int, horizon: int):
    defender = trainer.defender

    env = ADE()
    rewards = []
    for ep in range(episodes):
        env.reset()
        sampled_policy = sample_policy(defender.policies, sigma_D)
        reward = 0
        for k in range(horizon):

            s_d = env.state.uninvestigated_alerts

            d_action = run_defender_policy(sampled_policy, s_d)

            a_action = greedy_attacker(
                env.state.uninvestigated_alerts,
                env.state.attack_mounted,
                env.state.alerts_due_attack,
            )
            r_d, r_a = env.step(a_action, d_action)
            reward += r_d
        rewards.append(reward)

    return np.mean(rewards)


def greedy_attacker_vs_uniform_defender(episodes: int, horizon: int):
    env = ADE()

    rewards = []
    for ep in range(episodes):
        env.reset()
        reward = 0
        for k in range(horizon):
            d_action = uniform_policy(env.state.uninvestigated_alerts)
            a_action = greedy_attacker(
                env.state.uninvestigated_alerts,
                env.state.attack_mounted,
                env.state.alerts_due_attack,
            )

            r_d, r_a = env.step(a_action, d_action)
            reward += r_d

        rewards.append(reward)

    return np.mean(rewards)


if __name__ == "__main__":
    mc_reward = greedy_attacker_vs_uniform_defender(200, 50)
    print("Greedy attacker vs uniform defender:", mc_reward)
