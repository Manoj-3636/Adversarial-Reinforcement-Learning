import nashpy as nash
import numpy as np

def compute_nash_equilibrium(U):

    game = nash.Game(U, -U)

    equilibria = game.support_enumeration()

    for sigma_def, sigma_att in equilibria:
        return np.array(sigma_def), np.array(sigma_att)

    raise ValueError("No equilibrium found")