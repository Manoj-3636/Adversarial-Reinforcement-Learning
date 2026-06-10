# Prioritizing Alerts with Adversarial Reinforcement Learning

PyTorch implementation of the adversarial reinforcement learning framework introduced in **"Finding Needles in a Moving Haystack: Prioritizing Alerts with Adversarial Reinforcement Learning"** by Tong, Laszka, Yan, Zhang, and Vorobeychik.

## Table of Contents
- [Overview](#overview)
- [Problem](#problem)
- [Approach](#approach)
  - [Game-Theoretic Model](#game-theoretic-model)
  - [Double Oracle + DDPG-MIX](#double-oracle--ddpg-mix)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Attack Detection Environment](#attack-detection-environment)
  - [Hyperparameters](#hyperparameters)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Key Design Details](#key-design-details)
  - [State Spaces](#state-spaces)
  - [Action Spaces](#action-spaces)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Reward Function](#reward-function)
- [Results](#results)
- [Citation](#citation)

## Overview

Security detection systems (e.g., intrusion detection systems, fraud detectors) generate a massive number of alerts, the vast majority of which are false positives. Under limited investigation budgets, defenders must strategically prioritize which alerts to inspect. However, deterministic prioritization policies are easily exploited by adaptive attackers, who can simply choose attacks that trigger low-priority alerts.

This repository implements a **game-theoretic alert prioritization framework** that computes a **stochastic, dynamic, state-dependent defender policy** robust against a strong adaptive adversary. The adversary is assumed to have full knowledge of the system state and the defender's randomized policy, and can craft near-optimal attacks accordingly.

## Problem

- **Defender**: Has a limited budget to investigate alerts each time period. Observes only the current backlog of uninvestigated alerts (grouped by type).
- **Attacker**: Has a limited budget to mount attacks. Observes the full state — including which alerts exist, which attacks were previously executed, and which alerts they triggered. Also knows the defender's policy.
- **Goal**: Compute a defender policy that minimizes expected loss from undetected attacks, assuming the attacker plays a best response.

## Approach

### Game-Theoretic Model

The interaction is modeled as a **zero-sum stochastic game** played over discrete time periods:

1. **Defender investigates** a subset of existing alerts.
2. **Attacker observes** the full state and mounts attacks.
3. **Alerts are generated** — false alerts from benign traffic (Poisson) and true alerts from attacks.
4. **State updates** and the process repeats.

An attack is **detected** if *any* of the alerts it triggers is investigated in the current period. The defender's payoff is the negative sum of losses from all undetected attacks, discounted over time.

A **Mixed-Strategy Nash Equilibrium (MSNE)** of this game yields a robust stochastic policy: the defender randomizes over deterministic policies in a way that is unexploitable by the attacker.

### Double Oracle + DDPG-MIX

Because the policy spaces are combinatorial and intractably large, we use the **Double Oracle** algorithm combined with **Deep Deterministic Policy Gradient (DDPG-MIX)**:

1. **Maintain a restricted set of policies** for both players.
2. **Solve a zero-sum matrix game** via Linear Programming to obtain a provisional mixed-strategy Nash equilibrium.
3. **Train best-response oracles** using DDPG-MIX:
   - Each oracle learns an approximate optimal response to the opponent's current mixed strategy.
   - The opponent's mixed strategy is embedded into the environment by sampling a deterministic policy from their equilibrium distribution each episode.
4. **Add best responses to the restricted sets** and repeat until convergence.

This avoids enumerating the full policy space and allows gradient-based learning in continuous action spaces.

## Repository Structure

```
.
├── agents/
│   ├── ddpg.py               # Actor & Critic networks (PyTorch)
│   ├── attacker_agent.py     # Attacker DDPG agent + baselines (greedy, uniform)
│   └── defender_agent.py     # Defender DDPG agent + baselines (uniform, priority)
├── environment/
│   ├── config.py             # Domain parameters (attacks, alerts, costs, budgets, hyperparams)
│   ├── state_manager.py      # SystemState with hypergeometric detection probability
│   └── ade.py                # Attack Detection Environment (ADE) simulator
├── utils/
│   ├── lp_optimizer.py       # Zero-sum game LP solver (scipy.linprog)
│   ├── ri_utils.py           # ReplayBuffer & Epsilon-Greedy exploration
│   └── metrics.py            # TrainingLogger + learning curve plotting
├── train.py                  # Full Double Oracle training loop
├── evaluate.py               # Evaluation against baseline attackers
├── exports/
│   └── final_results.json    # Example output: equilibrium distributions & game value
├── pyproject.toml            # Python dependencies
└── uv.lock                   # Locked dependency versions
```

## Installation

This project uses **uv** for dependency management, but works with standard `pip` as well.

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

**Dependencies:**
- Python >= 3.10
- PyTorch
- NumPy
- SciPy
- Matplotlib

See `pyproject.toml` for exact versions.

## Configuration

All domain-specific parameters are centralized in `environment/config.py`.

### Attack Detection Environment

This implementation follows **Case Study I: Intrusion Detection** from the paper, using data from the **CICIDS2017 dataset** processed with the **Suricata IDS** (Emerging Threats Ruleset).

| Parameter | Value | Source |
|---|---|---|
| **Time period** | 30 minutes | Paper Section V-B |
| **Attacks (`A`)** | 7 types (Brute Force, Botnet, DoS, Heartbleed, Infiltration, PortScan, Web Attack) | Table IV |
| **Alert types (`T`)** | 7 types (e.g., `attempted-recon`, `attempted-user`, `bad-unknown`, ...) | Table III (pruned) |
| **True alert matrix** | Deterministic counts of alerts triggered per attack | Table IV |
| **False alert rates** | Poisson lambdas per alert type | Table V |
| **Attack costs (`E_a`)** | Minutes to mount attack | Table IV |
| **Defender losses (`L_a`)** | CVSS v3.0 base scores | Table IV |
| **Investigation cost (`C_t`)** | 1.0 (uniform) | Paper Section V-B |

### Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Discount factor (`τ`) | 0.95 | Paper Section V-A |
| Actor learning rate | 0.001 | Paper Section V-A |
| Critic learning rate | 0.002 | Paper Section V-A |
| Replay buffer size | 40,000 | Paper Section V-A |
| Actor hidden units | 32 | Table II (IDS) |
| Critic hidden units | 64 | Table II (IDS) |
| Defender default budget | 1000 | Paper Figure 4 |
| Attacker default budget | 120 | Paper Figure 4 |

You can override budgets and other parameters directly in `environment/config.py` to test robustness under different settings (e.g., defender budgets of 500 or 1500, attacker budgets of 60 or 180).

## Usage

### Training

Run the full Double Oracle training loop:

```bash
python train.py
```

This will:
1. Seed the policy pools with baseline policies (uniform/greedy attacker, uniform defender).
2. Iteratively build the utility matrix via Monte Carlo simulation.
3. Solve the LP to find the current mixed-strategy Nash equilibrium.
4. Train best-response policies for both players using DDPG-MIX.
5. Append best responses and repeat until convergence.

Output is saved to `exports/final_results.json`, containing:
- `defender_distribution`: Mixed strategy probabilities over defender policies
- `attacker_distribution`: Mixed strategy probabilities over attacker policies
- `game_value`: Expected defender payoff at equilibrium
- `value_history`: Convergence trace over iterations
- `defender_pool_size` / `attacker_pool_size`: Number of policies discovered

The training script also evaluates the final ARL defender against a greedy attacker baseline and prints the average defender loss.

### Evaluation

Standalone evaluation functions are provided in `evaluate.py`:

```bash
python evaluate.py
```

This compares defender policies against baseline attacker strategies (e.g., greedy attacker vs. uniform defender). You can also call `greedy_attacker_vs_arl_defender()` from `train.py` after training to evaluate the robustness of the learned mixed strategy.

## Key Design Details

### State Spaces

- **Attacker (full observability)**: Concatenation of:
  - `N`: uninvestigated alerts per type (`|T|`)
  - `M`: attacks mounted last round (`|A|`)
  - `S`: alerts triggered by each attack (`|A| × |T|`)
  - Total dimension: `|T| + |A| + |A|·|T|`

- **Defender (partial observability)**: Only `N` (uninvestigated alerts per type), log-transformed and normalized.

### Action Spaces

- **Attacker**: Outputs preference scores per attack. Enforced into a valid subset satisfying the budget constraint using a greedy cost-adjusted selection (ratio of score to cost).
- **Defender**: Outputs preference scores per alert type. Normalized into a probability distribution, multiplied by the budget, and clipped to available alert counts.

Both spaces are treated as **continuous** during RL training and discretized at execution time via budget projection, enabling efficient actor-critic learning.

### Neural Network Architecture

Following Table II of the paper (Intrusion Detection setting):

| Network | Hidden Layer | Units | Activation | Output | Init |
|---|---|---|---|---|---|
| **Actor** | Hidden | 32 | Tanh | Sigmoid (per action) | Xavier |
| **Critic** | Hidden | 64 | ReLU | Linear scalar | He Normal |

### Reward Function

- Defender reward per time step:
  ```
  R_def = -Σ_a (L_a · 1{attack a undetected})
  ```
- Detection probability for attack `a`: computed via the **hypergeometric distribution** — probability that the defender investigates *zero* alerts belonging to attack `a` when sampling `α_t` alerts from `N_t` total alerts (`S_{a,t}` of which belong to `a`).
- Attacker reward: `R_att = -R_def` (zero-sum).

## Results

An example training run produces equilibrium distributions similar to:

```json
{
  "defender_distribution": [0.0, 0.0, 0.0, 0.0, 0.920, 0.080, 0.0, 0.0],
  "attacker_distribution": [0.0, 0.033, 0.0, 0.0, 0.0, 0.0, 0.967, 0.0, 0.0],
  "game_value": -64.21,
  "value_history": [-18.67, -66.34, -66.07, -66.07, -64.05, -64.21, ...]
}
```

The Double Oracle typically converges in **< 15 iterations**, with the game value stabilizing as both players' best responses no longer improve upon the equilibrium.

As shown in the paper, the ARL approach:
- Outperforms **Uniform** and **Suricata priority-based** baselines by **~50%** in defender loss.
- Remains robust when the defender **misestimates the attacker budget** (< 5% degradation).
- Outperforms prior game-theoretic baselines (**GAIN**, **RIO**) that are restricted to static, single-period prioritization.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{tong2019finding,
  title={Finding Needles in a Moving Haystack: Prioritizing Alerts with Adversarial Reinforcement Learning},
  author={Tong, Liang and Laszka, Aron and Yan, Chao and Zhang, Ning and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:1906.08805},
  year={2019}
}
```
