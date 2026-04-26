import json
import os
import numpy as np
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

# ---------------------------------------------------------------------------
# Suricata priority policy
# Investigates alerts strictly by Suricata severity (1=highest) regardless
# of alert volume — matching the paper's Suricata baseline (Table III).
# Priority order: attempted-user, policy-violation, trojan-activity,
# unsuccessful-user, web-application-attack (p1), attempted-recon,
# bad-unknown (p2), misc-activity, not-suspicious,
# protocol-command-decode (p3)
# ---------------------------------------------------------------------------
SURICATA_PRIORITIES = np.array([2, 1, 2, 3, 3, 1, 3, 1, 1, 1])  # one per alert type


def suricata_policy(n):
    """
    Exhaust defense budget in ascending Suricata priority order
    (priority 1 first, then 2, then 3).
    """
    inv = np.zeros_like(n, dtype=np.int32)
    budget = DEFENDER_BUDGET_DEFAULT
    order = np.argsort(SURICATA_PRIORITIES)  # ascending → priority-1 alerts first
    for idx in order:
        take = int(min(n[idx], budget))
        inv[idx] = take
        budget -= take
        if budget == 0:
            break
    return inv


def uniform_attacker_policy(N, M, S):
    """
    Uniformly distributes the attacker budget across all attacks by randomly
    shuffling attack order and selecting each one until the budget runs out.
    Unlike greedy (which picks by loss/cost ratio) this ignores impact —
    every attack gets an equal chance of being selected each episode.
    """
    costs = np.array(list(ATTACK_COSTS.values()))
    action = np.zeros(len(costs), dtype=bool)
    budget = ATTACKER_BUDGET_DEFAULT

    # shuffle so no single attack is always favoured when budget runs short
    order = np.random.permutation(len(costs))

    for idx in order:
        if costs[idx] <= budget:
            action[idx] = True
            budget -= costs[idx]

    return action


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

# ---------------------------------------------------------------------------
# Budget enforcement helpers
# ---------------------------------------------------------------------------


def enforce_attacker_budget(
    raw_scores,
    costs=np.array(list(ATTACK_COSTS.values())),
    budget=ATTACKER_BUDGET_DEFAULT,
):
    """
    Greedily selects attacks by bang-per-buck (score/cost) until the budget
    is exhausted.

    raw_scores : preference scores from actor
    costs      : integer cost per attack
    budget     : total attacker resource budget
    returns    : bool vector – True where attack is executed
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
    """
    Converts continuous actor outputs to integer alert-investigation counts
    that respect the defender budget and the number of available alerts.

    raw_scores       : non-negative preference scores from actor (one per type)
    budget           : total investigation budget
    alerts_available : N^(k) – counts of uninvestigated alerts per type
    returns          : integer allocation vector
    """
    raw_scores = np.maximum(raw_scores, 0)

    if raw_scores.sum() == 0:
        raw_scores = np.ones_like(raw_scores)

    frac = raw_scores / raw_scores.sum()
    alloc = np.round(frac * budget).astype(np.int32)
    alloc = np.minimum(alloc, alerts_available)

    return alloc


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def run_episode(env, attacker_policy, defender_policy, horizon=50, discount=0.95):
    """
    Roll out one episode and return the *defender's* discounted reward.

    attacker_policy / defender_policy are Policy objects (func or neural-net).
    """
    env.reset()
    total = 0.0

    for t in range(horizon):
        d_action = run_defender_policy(
            defender_policy,
            env.state.uninvestigated_alerts,
        )
        a_action = run_attacker_policy(
            attacker_policy,
            env.state.uninvestigated_alerts,
            env.state.attack_mounted,
            env.state.alerts_due_attack,
        )
        r_d, _ = env.step(a_action, d_action)
        total += (discount**t) * r_d

    return total


def evaluate_attacker_vs_defenders(
    env,
    attacker_policy,
    defender_policies: dict,
    episodes=50,
    horizon=50,
    discount=0.95,
):
    """
    Evaluate one attacker policy against multiple named defender policies.

    defender_policies : {name: Policy}
    Returns           : {name: mean_defender_loss}
                        (loss = –reward, so higher is worse for the defender)
    """
    results = {}
    for name, def_policy in defender_policies.items():
        losses = []
        for _ in range(episodes):
            reward = run_episode(env, attacker_policy, def_policy, horizon, discount)
            losses.append(-reward)  # convert reward → loss
        results[name] = float(np.mean(losses))
    return results


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self):
        self.attacker = Attacker()
        self.defender = Defender()
        self.env = ADE()

        # ------------------------------------------------------------------ #
        # Seed policy pools with hand-crafted baselines                       #
        # ------------------------------------------------------------------ #

        # Attacker baselines
        self.attacker.policies.append(Policy("func", greedy_attacker, 0))
        self.attacker.policies.append(Policy("func", uniform_attacker, 0))

        # Defender baselines
        # uniform_policy  – allocates budget uniformly across alert types
        # priority_policy – mimics Suricata's built-in priority ordering
        self.defender.policies.append(Policy("func", uniform_policy, 0))
        self.defender.policies.append(Policy("func", priority_policy, 0))

    # ---------------------------------------------------------------------- #
    # Utility matrix (Monte-Carlo estimate)                                   #
    # ---------------------------------------------------------------------- #

    def build_utility_matrix(self, horizon=50, episodes=20):
        """
        Returns U[i, j] = E[discounted defender reward]
                          when defender uses policy i and attacker uses policy j.
        """
        m = len(self.defender.policies)
        n = len(self.attacker.policies)
        U = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                vals = []
                for _ in range(episodes):
                    self.env.reset()
                    total = 0.0
                    for t in range(horizon):
                        d_state = self.env.state.get_defender_state()
                        d_action = run_defender_policy(
                            self.defender.policies[i],
                            d_state.uninvestigated_alerts,
                        )
                        a_action = run_attacker_policy(
                            self.attacker.policies[j],
                            self.env.state.uninvestigated_alerts,
                            self.env.state.attack_mounted,
                            self.env.state.alerts_due_attack,
                        )
                        r_d, _ = self.env.step(a_action, d_action)
                        total += (0.95**t) * r_d
                    vals.append(total)
                U[i, j] = np.mean(vals)

        return U

    def sample_policy(self, pool, sigma):
        idx = np.random.choice(len(pool), p=sigma)
        return pool[idx]

    # ---------------------------------------------------------------------- #
    # Best-response training                                                   #
    # ---------------------------------------------------------------------- #

    def train_attacker_br(self, sigma_D, itr, episodes=200, horizon=50, batch_size=64):
        """
        Train an attacker best-response against the defender's mixed strategy
        sigma_D using DDPG-MIX.
        """
        learner = Attacker()
        memory = ReplayBuffer(100000)
        explore = EpsilonGreedyExploration(
            epsilon_start=0.30, epsilon_min=0.02, decay_rate=0.995
        )

        for ep in range(episodes):
            self.env.reset()
            opponent = self.sample_policy(self.defender.policies, sigma_D)

            for t in range(horizon):
                # State visible to the attacker: (N, M, S)
                s = preprocess_attacker_state(
                    self.env.state.uninvestigated_alerts,
                    self.env.state.attack_mounted,
                    self.env.state.alerts_due_attack,
                )

                with torch.no_grad():
                    raw = (
                        learner.policy(
                            torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                        )
                        .squeeze(0)
                        .numpy()
                    )

                raw = explore.select_action(raw, learner.action_dim)
                a_action = enforce_attacker_budget(raw)

                d_action = run_defender_policy(
                    opponent, self.env.state.uninvestigated_alerts
                )

                r_d, r_a = self.env.step(a_action, d_action)

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
        """
        Train a defender best-response against the attacker's mixed strategy
        sigma_A using DDPG-MIX.
        """
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

    # ---------------------------------------------------------------------- #
    # Attacker evaluation                                                     #
    # ---------------------------------------------------------------------- #

    def evaluate_uniform_attacker(
        self,
        arl_defender_policy,
        eval_episodes=100,
        horizon=50,
        export_dir="exports",
    ):
        """
        Test the uniform attacker against three defender policies:
          1. Uniform  – uniform budget allocation across alert types
          2. Suricata – strict Suricata severity ordering (Table III)
          3. ARL      – the trained neural-network defender

        Prints defender loss for each matchup.
        Returns: {defender_name: mean_defender_loss}
        """
        print("\n")
        print("=" * 60)
        print("ATTACKER EVALUATION  (Uniform Attacker)")
        print("=" * 60)

        defender_catalogue = {
            "Uniform": Policy("func", uniform_policy, 0),
            "Suricata": Policy("func", suricata_policy, 0),
            "ARL": arl_defender_policy,
        }

        uniform_att = Policy("func", uniform_attacker_policy, 0)

        results = evaluate_attacker_vs_defenders(
            self.env,
            uniform_att,
            defender_catalogue,
            episodes=eval_episodes,
            horizon=horizon,
        )

        print("\n  Uniform Attacker vs each defender:\n")
        print(f"  {'Defender':<12}  {'Mean Defender Loss':>20}")
        print("  " + "-" * 36)
        for def_name, loss in results.items():
            print(f"  {def_name:<12}  {loss:>20.4f}")
        print()
        print("  (Higher loss = attacker is more effective against that defender)")
        print("  (ARL defender should show the lowest loss)")

        os.makedirs(export_dir, exist_ok=True)
        out_path = os.path.join(export_dir, "attacker_evaluation.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")

        return results

    # ---------------------------------------------------------------------- #
    # Main double-oracle training loop                                        #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        iterations=20,
        matrix_episodes=20,
        matrix_horizon=50,
        br_episodes=200,
        br_horizon=50,
        export_dir="exports",
        tol=0.50,  # convergence threshold on game-value change
        patience=3,  # consecutive stable iterations required
        eval_episodes=100,  # episodes for final attacker evaluation
    ):
        os.makedirs(export_dir, exist_ok=True)

        values = []
        stable_count = 0

        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        for itr in range(1, iterations + 1):

            print("\n")
            print("=" * 60)
            print(f"DOUBLE ORACLE ITERATION {itr}")
            print("=" * 60)

            # ============================================================== #
            # 1. BUILD UTILITY MATRIX                                         #
            # ============================================================== #
            print("Building Utility Matrix...")
            U = self.build_utility_matrix(
                horizon=matrix_horizon, episodes=matrix_episodes
            )
            print("Utility Matrix Built:")
            print(U)

            # ============================================================== #
            # 2. LP SOLVE – obtain mixed-strategy Nash equilibrium            #
            # ============================================================== #
            print("\nSolving LP Equilibrium...")
            sigma_D, sigma_A, value = solve_zero_sum_game(U)

            print("LP Solved.")
            print("Defender Distribution:")
            print(np.round(sigma_D, 4))
            print("Attacker Distribution:")
            print(np.round(sigma_A, 4))
            print("Game Value:")
            print(value)

            values.append(value)

            # ============================================================== #
            # 3. CHECK CONVERGENCE                                            #
            # ============================================================== #
            if len(values) >= 2:
                diff = abs(values[-1] - values[-2])
                print(f"\nChange from previous value: {diff:.4f}")

                if diff < tol:
                    stable_count += 1
                    print(f"Stable iterations: {stable_count}/{patience}")
                else:
                    stable_count = 0

                if stable_count >= patience:
                    print("\nConvergence reached – stopping early.")
                    break

            # ============================================================== #
            # 4. TRAIN ATTACKER BEST RESPONSE                                 #
            # ============================================================== #
            print("\nTraining Attacker Best Response...")
            new_att = self.train_attacker_br(
                sigma_D, itr=itr, episodes=br_episodes, horizon=br_horizon
            )

            # ============================================================== #
            # 5. TRAIN DEFENDER BEST RESPONSE                                 #
            # ============================================================== #
            print("\nTraining Defender Best Response...")
            new_def = self.train_defender_br(
                sigma_A, itr=itr, episodes=br_episodes, horizon=br_horizon
            )

            # ============================================================== #
            # 6. EXPAND POLICY POOLS                                          #
            # ============================================================== #
            self.attacker.policies.append(new_att)
            print(f"Attacker BR added. Pool size: {len(self.attacker.policies)}")

            self.defender.policies.append(new_def)
            print(f"Defender BR added. Pool size: {len(self.defender.policies)}")

        # ================================================================== #
        # FINAL EQUILIBRIUM                                                   #
        # ================================================================== #
        print("\n")
        print("=" * 60)
        print("FINAL EXPORT")
        print("=" * 60)

        print("Computing final equilibrium...")
        U = self.build_utility_matrix(horizon=matrix_horizon, episodes=matrix_episodes)
        sigma_D, sigma_A, value = solve_zero_sum_game(U)

        print("Done.")
        print("\nFinal Defender Distribution:")
        print(np.round(sigma_D, 4))
        print("\nFinal Attacker Distribution:")
        print(np.round(sigma_A, 4))
        print(f"\nFinal Game Value: {value:.4f}")

        # ------------------------------------------------------------------ #
        # Retrieve the ARL defender to use in attacker evaluation            #
        # The equilibrium mixed strategy assigns weights over all policies;  #
        # we select the neural-net policy with the highest sigma_D weight.   #
        # ------------------------------------------------------------------ #
        arl_defender_policy = self._get_best_arl_defender(sigma_D)

        # ================================================================== #
        # ATTACKER EVALUATION                                                 #
        # Test uniform attacker vs Uniform / Suricata / ARL defenders        #
        # ================================================================== #
        attacker_eval = self.evaluate_uniform_attacker(
            arl_defender_policy=arl_defender_policy,
            eval_episodes=eval_episodes,
            horizon=br_horizon,
            export_dir=export_dir,
        )

        # ------------------------------------------------------------------ #
        # Save all results                                                    #
        # ------------------------------------------------------------------ #
        final_info = {
            "defender_distribution": sigma_D.tolist(),
            "attacker_distribution": sigma_A.tolist(),
            "game_value": float(value),
            "value_history": values,
            "defender_pool_size": len(self.defender.policies),
            "attacker_pool_size": len(self.attacker.policies),
            "attacker_evaluation": attacker_eval,
        }

        out_path = os.path.join(export_dir, "final_results.json")
        with open(out_path, "w") as f:
            json.dump(final_info, f, indent=2)

        print(f"\nAll results exported to: {export_dir}")

        return sigma_D, sigma_A, value

    def _get_best_arl_defender(self, sigma_D):
        """
        Select the ARL (neural-net) defender policy that carries the highest
        weight in the equilibrium mixed strategy sigma_D.

        Falls back to the uniform baseline if no neural policy has been
        trained.
        """
        best_policy = None
        best_weight = -1.0

        for idx, policy in enumerate(self.defender.policies):
            if policy.kind != "func" and sigma_D[idx] > best_weight:
                best_weight = sigma_D[idx]
                best_policy = policy

        if best_policy is None:
            print(
                "  [warn] No trained ARL defender found – "
                "falling back to Uniform baseline for evaluation."
            )
            return Policy("func", uniform_policy, 0)

        return best_policy


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(
        matrix_episodes=20,
        matrix_horizon=50,
        br_episodes=200,
        br_horizon=50,
        iterations=20,
        eval_episodes=100,
    )
