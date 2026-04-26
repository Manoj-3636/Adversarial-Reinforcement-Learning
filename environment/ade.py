from environment.state_manager import SystemState
import numpy as np
from numpy.typing import NDArray
from environment.config import (
    ALERTS,ATTACKS
)

class ADE:
    def __init__(self):
        self.state = SystemState(
            uninvestigated_alerts=np.zeros(len(ALERTS), dtype=np.int32),
            attack_mounted=np.zeros(len(ATTACKS), dtype=np.bool_),
            alerts_due_attack=np.zeros((len(ATTACKS), len(ALERTS)), dtype=np.int32)
        )

    def step(self,attack_action:NDArray,defense_action:NDArray):
        defense_action = defense_action.clip(np.zeros_like(defense_action),self.state.uninvestigated_alerts)
        defense_action = defense_action.astype(np.int32)
        # Do the defender action first there will be no alerts in the first iteration but its fine
        # Don't reset after every step
        reward_def, reward_att = self.state.state_update_defender(defense_action)
        self.state.generateFalseAlerts()
        self.state.state_update_attacker(attack_action)

        return reward_def,reward_att

    def reset(self):
        self.state.reset()