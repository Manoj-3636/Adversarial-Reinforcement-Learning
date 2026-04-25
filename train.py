import numpy as np
from environment.state_manager import SystemState
from environment.config import ALERTS, ATTACKS, DEFENDER_BUDGET_DEFAULT, RL_PARAMS
from agents.agent_wrapper import DDPG_Defender
from utils.ri_utils import ReplayBuffer, EpsilonGreedyExploration
from utils.metrics import TrainingLogger


def get_normalized_state(uninvestigated_alerts):
    """
    Neural nets struggle with large numbers (like 44,100 alerts).
    We normalize the state observation using log scale to keep values stable.
    """
    return np.log1p(uninvestigated_alerts.astype(np.float32))


def map_action_to_budget(raw_action, budget):
    """Maps the NN's continuous [0,1] output to an integer investigation array"""
    # Normalize the output so it sums to 1
    action_probs = raw_action / (np.sum(raw_action) + 1e-8)
    # Distribute the budget
    investigation_array = np.floor(action_probs * budget).astype(np.int32)
    return investigation_array


def sample_opponent_mixed_strategy():
    """Line 5: Simulate the opponent (Attacker) picking a strategy"""
    # For now, let's assume the attacker uniformly randomly picks one attack type.
    # Later, you can swap this with Nash Equilibrium probabilities from lp_optimizer.py
    attack_idx = np.random.randint(0, len(ATTACKS))
    attack_array = np.zeros(len(ATTACKS), dtype=bool)
    attack_array[attack_idx] = True
    return attack_array


def train():
    state_dim = len(ALERTS)
    action_dim = len(ALERTS)

    # Initialize classes from your repo
    agent = DDPG_Defender(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=RL_PARAMS["replay_buffer_size"])
    logger = TrainingLogger(case_name="Intrusion Detection")

    explorer = EpsilonGreedyExploration()
    explorer.__init__(epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995)

    M_EPISODES = RL_PARAMS["training_episodes"]
    K_STEPS = RL_PARAMS["steps_per_episode"]
    BUDGET = DEFENDER_BUDGET_DEFAULT

    print("Starting Training...")

    # Line 3: Outer Episode Loop
    for episode in range(M_EPISODES):

        # Line 4: Initialize System State (Blank slate)
        system = SystemState(
            uninvestigated_alerts=np.zeros(len(ALERTS), dtype=np.int32),
            attack_mounted=np.zeros(len(ATTACKS), dtype=bool),
            alerts_due_attack=np.zeros((len(ATTACKS), len(ALERTS)), dtype=np.int32)
        )

        # Line 5: Sample opponent's policy for this episode
        opponent_attack_action = sample_opponent_mixed_strategy()

        episode_loss_tracker = 0

        # Line 6: Inner Step Loop
        for k in range(K_STEPS):
            # Current State Observation
            current_obs = get_normalized_state(system.uninvestigated_alerts)

            # Line 7: Select Action
            raw_action = agent.get_action(current_obs, explorer)
            investigate_array = map_action_to_budget(raw_action, BUDGET)

            # Line 8: Execute Actions and Transit State
            # 1. Attacker attacks (generates real + false alerts)
            system.state_update_attacker(opponent_attack_action)

            # 2. Defender investigates
            reward_def, reward_att = system.state_update_defender(investigate_array)

            # 3. Observe new state
            next_obs = get_normalized_state(system.uninvestigated_alerts)

            # Line 9: Store Transition
            replay_buffer.push(current_obs, raw_action, reward_def, next_obs)

            # Line 10-13: Update Networks
            agent.train_step(replay_buffer, batch_size=64)

            # Note: The defender's "loss" is mathematically negative reward.
            episode_loss_tracker += abs(reward_def)

            # End of Episode routines
        explorer.decay()
        logger.add_step_loss(episode_loss_tracker)
        logger.end_episode()

        if episode % 10 == 0:
            avg_loss = logger.get_average_episode_loss(window=10)
            print(f"Episode {episode} | Avg Loss (last 10): {avg_loss:.2f} | Epsilon: {explorer.epsilon:.3f}")

    # Plot results using your metrics.py class
    logger.plot_learning_curve("learning_curve.png")
    print("Training Complete. Model trained and learning curve saved.")


if __name__ == "__main__":
    train()