from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from environment.config import ALERTS, ATTACKS, TRUE_ALERT_MATRIX
from scipy.stats import hypergeom


def _calculate_undetected_prob(N_vec:NDArray[np.int32], S_matrix:NDArray[np.int32], alpha_vec:NDArray[np.int32],M_vec:NDArray[np.int32]):


    # 1. Mask out alerts where N=0 to avoid division-by-zero / NaNs
    valid_mask = N_vec > 0

    # Initialize a probability matrix with 1.0 (100% survival)
    # Shape: (num_attacks, num_alerts)
    prob_matrix = np.ones_like(S_matrix, dtype=np.float64)

    # 2. Compute probabilities across the whole 2D grid at once.
    # NumPy automatically broadcasts the 1D N_vec and alpha_vec across the rows of S_matrix.
    prob_matrix[:, valid_mask] = hypergeom.pmf(
        0,  # Exactly 0 attack alerts drawn
        N_vec[valid_mask],  # Broadcasts to (num_attacks, valid_alerts)
        S_matrix[:, valid_mask],  # Sliced to (num_attacks, valid_alerts)
        alpha_vec[valid_mask]  # Broadcasts to (num_attacks, valid_alerts)
    )

    # 3. Multiply probabilities across the alert types (axis=1)
    # This gives a 1D array of length `num_attacks`, representing p_a for each attack
    p_a_vec = np.prod(prob_matrix, axis=1)

    # 4. Zero out the probabilities for attacks that were never mounted this round
    return p_a_vec * M_vec

@dataclass(
    slots=True
)
class SystemState:
    # recon, user, unknown, misc, not-susp, policy, protocol
    uninvestigated_alerts:NDArray[np.int32]
    attack_mounted:NDArray[np.bool_]
    alerts_due_attack:NDArray[np.int32]

    def __post_init__(self):
        assert self.uninvestigated_alerts.shape == (len(ALERTS),)
        assert self.attack_mounted.shape == (len(ATTACKS),)
        assert self.alerts_due_attack.shape == (len(ATTACKS), len(ALERTS))

    def state_update_attacker(self,attack_array:NDArray[np.bool_]):
        """
        Updates the state based on the given attacker's action
        :param attack_array: binary array representing the attacker's attack
        :return: None
        """
        self.attack_mounted = attack_array
        self.uninvestigated_alerts += TRUE_ALERT_MATRIX[self.attack_mounted]
        self.alerts_due_attack += TRUE_ALERT_MATRIX[self.attack_mounted]
        return self.attack_mounted,self.uninvestigated_alerts,self.alerts_due_attack

    def state_update_defender(self,investigation_array:NDArray[np.int32]):
        """
        Updates the state based on the given defender's investigation array
        :param investigation_array: array representing how many alerts of each type
               are investigated
        :return:None
        """



    def step(self):
        """
        Give the reward for the defender on each step
        :return: Reward for defender
        """





    def _get_defender_state(self):
        return DefenderState(uninvestigated_alerts=self.uninvestigated_alerts)


@dataclass(
    slots=True
)
class DefenderState:
    uninvestigated_alerts: np.ndarray

    def __post_init__(self):
        assert self.uninvestigated_alerts.shape == (len(ALERTS),)


