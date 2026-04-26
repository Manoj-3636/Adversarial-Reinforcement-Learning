import numpy as np
from scipy.optimize import linprog


def solve_zero_sum_game(U):
    """
    U[i,j] = defender payoff
    rows   = defender strategies
    cols   = attacker strategies

    Returns:
        sigma_def
        sigma_att
        value
    """

    U = np.array(U, dtype=float)

    m, n = U.shape

    # ---------- Defender LP ----------
    # variables:
    # x0..x(m-1) probabilities
    # v game value
    #
    # maximize v
    # s.t.
    #   sum_i x_i U[i,j] >= v   for all j
    #   sum_i x_i = 1
    #   x_i >= 0

    # linprog minimizes, so minimize -v

    c = np.zeros(m + 1)
    c[-1] = -1.0

    A_ub = []
    b_ub = []

    for j in range(n):
        row = np.zeros(m + 1)
        row[:m] = -U[:, j]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = [np.ones(m + 1)]
    A_eq[0][-1] = 0.0

    b_eq = [1.0]

    bounds = [(0, None)] * m + [(None, None)]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if not res.success:
        raise ValueError("Defender LP failed")

    sigma_def = res.x[:m]
    value = res.x[-1]

    # ---------- Attacker LP ----------
    # minimize w
    # s.t.
    # sum_j y_j U[i,j] <= w   all i
    # sum_j y_j =1

    c2 = np.zeros(n + 1)
    c2[-1] = 1.0

    A_ub2 = []
    b_ub2 = []

    for i in range(m):
        row = np.zeros(n + 1)
        row[:n] = U[i, :]
        row[-1] = -1.0
        A_ub2.append(row)
        b_ub2.append(0.0)

    A_eq2 = [np.ones(n + 1)]
    A_eq2[0][-1] = 0.0

    b_eq2 = [1.0]

    bounds2 = [(0, None)] * n + [(None, None)]

    res2 = linprog(
        c2,
        A_ub=A_ub2,
        b_ub=b_ub2,
        A_eq=A_eq2,
        b_eq=b_eq2,
        bounds=bounds2,
        method="highs",
    )

    if not res2.success:
        raise ValueError("Attacker LP failed")

    sigma_att = res2.x[:n]

    sigma_def = np.clip(sigma_def, 0, None)
    sigma_def /= sigma_def.sum()

    sigma_att = np.clip(sigma_att, 0, None)
    sigma_att /= sigma_att.sum()

    return sigma_def, sigma_att, value
