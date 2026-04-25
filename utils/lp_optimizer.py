import nashpy as nash

def compute_nash_equilibrium(utility_matrix):

    game = nash.Game(utility_matrix)
    equilibria = game.support_enumeration()

    for eq in equilibria :
        sigma_def, sigma_att = eq
        return sigma_def, sigma_att

