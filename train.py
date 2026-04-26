import json
import os

from agents.attacker_agent import (
    Attacker,
    greedy_attacker,
    run_attacker_policy,
    uniform_attacker,
    preprocess_attacker_state,
)
from agents.ddpg import Policy
from agents.defender_agent import (
    Defender,
    uniform_policy,
    run_defender_policy,
    priority_policy,
    preprocess_defender_state,
)
from environment.ade import ADE
import numpy as np
import torch

from environment.config import (
    ATTACK_COSTS,
    ATTACKER_BUDGET_DEFAULT,
    DEFENDER_BUDGET_DEFAULT,
)
from utils.lp_optimizer import solve_zero_sum_game
from utils.ri_utils import ReplayBuffer, EpsilonGreedyExploration


def sample_policy(self, pool, sigma):
    idx = np.random.choice(len(pool), p=sigma)

    return pool[idx]


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


def enforce_defender_budget(raw_scores, budget, alerts_available):
    raw_scores = np.maximum(raw_scores, 0)

    if raw_scores.sum() == 0:
        raw_scores = np.ones_like(raw_scores)

    frac = raw_scores / raw_scores.sum()

    alloc = np.round(frac * budget).astype(np.int32)

    alloc = np.minimum(alloc, alerts_available)

    return alloc


class Trainer:
    def __init__(self):
        self.attacker = Attacker()
        self.defender = Defender()
        self.env = ADE()

        # Seed with initial policies
        self.attacker.policies.append(Policy("func", greedy_attacker, 0))
        self.attacker.policies.append(Policy("func", uniform_attacker, 0))

        self.defender.policies.append(Policy("func", uniform_policy, 0))
        self.defender.policies.append(Policy("func", priority_policy, 0))

    # build using an MC estimate
    def build_utility_matrix(self, horizon=50, episodes=20):
        m = len(self.defender.policies)
        n = len(self.attacker.policies)

        U = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                vals = []

                for ep in range(episodes):

                    state = self.env.reset()
                    total = 0.0

                    for t in range(horizon):
                        d_state = self.env.state.get_defender_state()
                        d_action = run_defender_policy(
                            self.defender.policies[i], d_state.uninvestigated_alerts
                        )
                        a_action = run_attacker_policy(
                            self.attacker.policies[j],
                            self.env.state.uninvestigated_alerts,
                            self.env.state.attack_mounted,
                            self.env.state.alerts_due_attack,
                        )

                        r_d, r_a = self.env.step(a_action, d_action)
                        total += (0.95**t) * r_d
                    vals.append(total)

                U[i, j] = np.mean(vals)
        return U

    def sample_policy(self, pool, sigma):
        idx = np.random.choice(len(pool), p=sigma)

        return pool[idx]

    def train_attacker_br(self, sigma_D, itr, episodes=200, horizon=50, batch_size=64):
        learner = Attacker()
        memory = ReplayBuffer(100000)
        explore = EpsilonGreedyExploration(
            epsilon_start=0.30, epsilon_min=0.02, decay_rate=0.995
        )
        for ep in range(episodes):
            self.env.reset()
            opponent = self.sample_policy(self.defender.policies, sigma_D)
            for t in range(horizon):

                # -------------------------
                # state = (N,M,S)
                # -------------------------
                s = preprocess_attacker_state(
                    self.env.state.uninvestigated_alerts,
                    self.env.state.attack_mounted,
                    self.env.state.alerts_due_attack,
                )
                # actor output
                with torch.no_grad():

                    raw = (
                        learner.policy(
                            torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                        )
                        .squeeze(0)
                        .numpy()
                    )

                # exploration
                raw = explore.select_action(raw, learner.action_dim)

                # -------------------------
                # enforce resource budget
                # -------------------------
                a_action = enforce_attacker_budget(raw)

                # defender move
                d_action = run_defender_policy(
                    opponent, self.env.state.uninvestigated_alerts
                )

                # env step
                r_d, r_a = self.env.step(a_action, d_action)

                # next state
                s2 = preprocess_attacker_state(
                    self.env.state.uninvestigated_alerts,
                    self.env.state.attack_mounted,
                    self.env.state.alerts_due_attack,
                )

                memory.push(s, a_action.astype(np.float32), r_a, s2)

                if len(memory) >= batch_size:
                    batch = memory.sample(batch_size)

                    learner.update(*batch)

            explore.decay()

        learner.save_policy(itr)

        return learner.policies[-1]

    def train_defender_br(self, sigma_A, itr, episodes=200, horizon=50, batch_size=64):

        learner = Defender()

        memory = ReplayBuffer(100000)

        explore = EpsilonGreedyExploration(
            epsilon_start=0.25, epsilon_min=0.02, decay_rate=0.995
        )

        for ep in range(episodes):

            self.env.reset()

            opponent = self.sample_policy(self.attacker.policies, sigma_A)

            for t in range(horizon):

                N = self.env.state.uninvestigated_alerts

                s = preprocess_defender_state(N)

                with torch.no_grad():

                    raw = (
                        learner.policy(
                            torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                        )
                        .squeeze(0)
                        .numpy()
                    )

                raw = explore.select_action(raw, learner.action_dim)

                d_action = enforce_defender_budget(raw, DEFENDER_BUDGET_DEFAULT, N)

                a_action = run_attacker_policy(
                    opponent,
                    self.env.state.uninvestigated_alerts,
                    self.env.state.attack_mounted,
                    self.env.state.alerts_due_attack,
                )

                r_d, r_a = self.env.step(a_action, d_action)

                s2 = preprocess_defender_state(self.env.state.uninvestigated_alerts)

                memory.push(s, d_action.astype(np.float32), r_d, s2)

                if len(memory) >= batch_size:
                    batch = memory.sample(batch_size)
                    learner.update(*batch)

            explore.decay()

        learner.save_policy(itr)

        return learner.policies[-1]

    def train(
        self,
        iterations=10,
        matrix_episodes=20,
        matrix_horizon=30,
        br_episodes=200,
        br_horizon=50,
        export_dir="exports",
    ):

        os.makedirs(export_dir, exist_ok=True)

        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        for itr in range(1, iterations + 1):
            print("\n")
            print("=" * 60)
            print(f"DOUBLE ORACLE ITERATION {itr}")
            print("=" * 60)

            print("Building Utility Matrix...")

            U = self.build_utility_matrix(
                horizon=matrix_horizon, episodes=matrix_episodes
            )

            print("Utility Matrix Built:")
            print(U)

            print("\nSolving LP Equilibrium...")

            sigma_D, sigma_A, value = solve_zero_sum_game(U)

            print("LP Solved.")

            print("Defender Distribution:")
            print(np.round(sigma_D, 4))

            print("Attacker Distribution:")
            print(np.round(sigma_A, 4))

            print("Game Value:")
            print(value)

            print("\nTraining Attacker Best Response...")

            new_att = self.train_attacker_br(
                sigma_D, itr=itr, episodes=br_episodes, horizon=br_horizon
            )

            print("\nTraining Defender Best Response...")

            new_def = self.train_defender_br(
                sigma_A, itr=itr, episodes=br_episodes, horizon=br_horizon
            )

            self.attacker.policies.append(new_att)
            print("Attacker BR Added.")
            print("Attacker Pool Size:", len(self.attacker.policies))
            self.defender.policies.append(new_def)

            print("Defender BR Added.")
            print("Defender Pool Size:", len(self.defender.policies))

        # ======================================================
        # FINAL EXPORT
        # ======================================================
        print("\n")
        print("=" * 60)
        print("FINAL EXPORT")
        print("=" * 60)

        print("Computing Final Equilibrium...")

        U = self.build_utility_matrix(horizon=matrix_horizon, episodes=matrix_episodes)

        sigma_D, sigma_A, value = solve_zero_sum_game(U)

        print("Done.")

        print("\nFinal Defender Distribution:")
        print(np.round(sigma_D, 4))

        print("\nFinal Attacker Distribution:")
        print(np.round(sigma_A, 4))

        print("\nFinal Value:")
        print(value)

        # ------------------------------------------------------
        # save defender pool
        # ------------------------------------------------------
        os.makedirs(f"{export_dir}/defender_pool", exist_ok=True)

        for i, pol in enumerate(self.defender.policies):

            if pol.type == "nn":
                torch.save(pol.model, f"{export_dir}/defender_pool/policy_{i}.pt")

        # ------------------------------------------------------
        # save attacker pool
        # ------------------------------------------------------
        os.makedirs(f"{export_dir}/attacker_pool", exist_ok=True)

        for i, pol in enumerate(self.attacker.policies):

            if pol.type == "nn":
                torch.save(pol.model, f"{export_dir}/attacker_pool/policy_{i}.pt")

        # ------------------------------------------------------
        # save distributions
        # ------------------------------------------------------
        final_info = {
            "defender_distribution": sigma_D.tolist(),
            "attacker_distribution": sigma_A.tolist(),
            "game_value": float(value),
            "defender_pool_size": len(self.defender.policies),
            "attacker_pool_size": len(self.attacker.policies),
        }

        with open(f"{export_dir}/final_results.json", "w") as f:
            json.dump(final_info, f, indent=2)

        print("\nEverything Exported To:")
        print(export_dir)

        return sigma_D, sigma_A, value


if __name__ == "__main__":
    trainer = Trainer()
    # matrix = trainer.build_utility_matrix()
    # print(matrix)
    # print(solve_zero_sum_game(matrix))

    trainer.train(matrix_episodes=20, matrix_horizon=50)
