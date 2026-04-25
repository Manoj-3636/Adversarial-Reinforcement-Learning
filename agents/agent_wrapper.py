import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from agents.ddpg import Actor, Critic
from environment.config import RL_PARAMS


class DDPG_Defender:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Networks (Line 1)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=RL_PARAMS["actor_learning_rate"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=RL_PARAMS["critic_learning_rate"])

        self.gamma = RL_PARAMS["discount_factor_tau"]
        self.action_dim = action_dim

    def get_action(self, state, eps_greedy_obj):
        """Line 7: Epsilon-Greedy Exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get raw policy output [0, 1] from Sigmoid
            policy_action = self.actor(state_tensor).cpu().numpy()[0]

        # Apply your EpsilonGreedyExploration from ri_utils
        action = eps_greedy_obj.select_action(policy_action, self.action_dim)
        return action

    def train_step(self, replay_buffer, batch_size=64):
        """Lines 10-13: Sample and Update"""
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # ------------------- UPDATE CRITIC (Line 11 & 12) -------------------
        with torch.no_grad():
            next_actions = self.actor(next_states)
            # Target Q-Value
            target_q = rewards + self.gamma * self.critic(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------- UPDATE ACTOR (Line 13) -------------------------
        predicted_actions = self.actor(states)
        # Minimize -Q to Maximize Q
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()