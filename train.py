from agents.attacker_agent import Attacker, greedy_attacker, run_attacker_policy, uniform_attacker
from agents.ddpg import Policy
from agents.defender_agent import Defender, uniform_policy, run_defender_policy, priority_policy
from environment.ade import ADE
import numpy as np

class Trainer:
    def __init__(self):
        self.attacker = Attacker()
        self.defender = Defender()
        self.env = ADE()

        # Seed with initial policies
        self.attacker.policies.append(Policy("func",greedy_attacker,0))
        self.attacker.policies.append(Policy("func",uniform_attacker,0))

        self.defender.policies.append(Policy("func",uniform_policy,0))
        self.defender.policies.append(Policy("func",priority_policy,0))


    # build using an MC estimate
    def build_utility_matrix(self, horizon=50, episodes=20):
        m = len(self.defender.policies)
        n = len(self.attacker.policies)

        U = np.zeros((m,n))

        for i in range(m):
            for j in range(n):
                vals = []

                for ep in range(episodes):

                    state = self.env.reset()
                    total = 0.0

                    for t in range(horizon):
                        d_state = self.env.state.get_defender_state()
                        d_action = run_defender_policy(self.defender.policies[i], d_state.uninvestigated_alerts)
                        a_action = run_attacker_policy(self.attacker.policies[i],self.env.state.uninvestigated_alerts,self.env.state.alerts_due_attack)

                        r_d,r_a = self.env.step(a_action,d_action)
                        total += (0.95 ** t) * r_d
                    vals.append(total)

                U[i,j] = np.mean(vals)
        return U



if __name__ == "__main__":
    trainer = Trainer()
    print(trainer.build_utility_matrix())

