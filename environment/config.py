"""
config.py

Global parameters, costs, budgets, and probability distributions for the
Adversarial Reinforcement Learning Alert Prioritization Environment.
Data extracted from: "Finding Needles in a Moving Haystack" (Case Study I)
"""

import numpy as np

# ==========================================
# 1. Action Space (Attacker)
# ==========================================
# List of representative attacks (from Table IV)
ATTACKS = [
    "Brute Force",
    "Botnet",
    "DoS",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "Web Attack"
]

# Attack Execution Costs (E_a) - Time in minutes to mount the attack
ATTACK_COSTS = {
    "Brute Force": 120.0,
    "Botnet": 60.0,
    "DoS": 74.0,
    "Heartbleed": 20.0,
    "Infiltration": 52.0,
    "PortScan": 80.0,
    "Web Attack": 62.0
}

# Defender Loss (L_a) - Incurred if attack is undetected (Based on CVSS v3.0)
# Note: 'Web Attack' lists 27 in the text, though CVSS maxes at 10.0.
# Keeping as 27.0 to strictly match the paper's Table IV.
ATTACK_LOSSES = {
    "Brute Force": 3.6,
    "Botnet": 6.0,
    "DoS": 4.0,
    "Heartbleed": 3.6,
    "Infiltration": 1.4,
    "PortScan": 1.4,
    "Web Attack": 2.7
}

# ==========================================
# 2. Observation Space (Defender Alerts)
# ==========================================
# List of pruned alert types (from Table III & Table V)
ALERTS = [
    "attempted-recon",
    "attempted-user",
    "bad-unknown",
    "misc-activity",
    "not-suspicious",
    "policy-violation",
    "protocol-command-decode"
]

# Cost to investigate a single alert (C_t)
# "We set the cost of investigating each alert to 1.0 (i.e., equal for all alerts)."
INVESTIGATION_COST_PER_ALERT = 1.0

# ==========================================
# 3. Environment Dynamics (Alert Generation)
# ==========================================

# True Positives: Number of each alert type raised per attack (Table IV)
# The paper treats these as deterministic counts once an attack is mounted.
# Shape: (len(ATTACKS), len(ALERTS))
TRUE_ALERT_MATRIX = np.array([
    # recon, user, unknown, misc, not-susp, policy, protocol
    [1230, 0, 0, 0, 0, 0, 0],         # Brute Force
    [0, 4, 2, 106, 0, 54, 0],         # Botnet
    [0, 0, 0, 0, 0, 24, 0],           # DoS
    [0, 0, 4, 0, 10, 0, 0],           # Heartbleed
    [710, 2, 862, 12, 0, 80, 600],    # Infiltration
    [138, 0, 320, 30, 0, 0, 0],       # PortScan
    [0, 0, 6, 10, 0, 0, 0]            # Web Attack
], dtype=np.int32)

# False Positives: Benign traffic alert generation (Table V)
# These are the means (lambda) for the Poisson distributions.
# Represents the average number of false alerts triggered in each 30-min period.
FALSE_ALERT_LAMBDAS = {
    "attempted-recon": 7200.0,
    "attempted-user": 44100.0,
    "bad-unknown": 1600.0,
    "misc-activity": 7300.0,
    "not-suspicious": 17400.0,
    "policy-violation": 4000.0,
    "protocol-command-decode": 10200.0
}

# Convert lambda dict to a numpy array matching the order of ALERTS for fast simulation
FALSE_ALERT_LAMBDA_ARRAY = np.array([
    FALSE_ALERT_LAMBDAS[alert] for alert in ALERTS
], dtype=np.float32)

# ==========================================
# 4. Global Budgets and RL Hyperparameters
# ==========================================

# Budgets evaluated in the paper (from Figure 4)
# You can override these in your training loop to test robustness
DEFENDER_BUDGET_DEFAULT = 1000.0  # Paper tests [500, 1000, 1500]
ATTACKER_BUDGET_DEFAULT = 120.0   # Paper tests [60, 120, 180]

# Time period definition (for logging/context)
TIME_PERIOD_MINUTES = 30

# DDPG-MIX Hyperparameters (from Section V.A.1)
RL_PARAMS = {
    "discount_factor_tau": 0.95,
    "actor_learning_rate": 0.001,
    "critic_learning_rate": 0.002,
    "replay_buffer_size": 40000,
    "training_episodes": 500,
    "steps_per_episode": 400
}

# Neural Network Architecture details (Table II - Intrusion Detection sizes)
NN_ARCH = {
    "actor_hidden_units": 32,
    "critic_hidden_units": 64,
    "actor_activation": "Tanh",      # Output layer uses Sigmoid
    "critic_activation": "Relu"
}