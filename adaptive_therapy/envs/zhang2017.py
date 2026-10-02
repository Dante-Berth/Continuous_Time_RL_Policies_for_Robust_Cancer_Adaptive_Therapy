"""Zhang 2017 three-type competition: the model the abiraterone AT trial used.

    dx_i/dt = r_i x_i (1 - sum_j a_ij x_j / K_i),   i = T+, TP, T-
    dPSA/dt = sum_i x_i - 0.5 PSA

T+ cells need androgen and "cheat" on the testosterone TP cells make, so their
capacity is tied to TP: K_T+ = 1.5 x_TP off drug, 0.5 x_TP on. Abiraterone cuts
K_TP from 10,000 to 100. T- cells are androgen-independent and resistant,
K_T- = 10,000 throughout. Resistance therefore acts through producer-dependent
capacities, not through a drug-suppressed growth term as in our two-type LV.
That is why this family is the program's second resistance mechanism (ROADMAP
V5).

Equations and values: Zhang et al. Nat Commun 2017;8:1816, Eq. 1-2 and Methods
p.7-8, Table 3 p.8 (read 1 deg, literature/papers/url-zhang-2017-mcrpc-at.md).
"""

from __future__ import annotations

import jax.numpy as jnp

# theta layout, fixed: growth rates, the six off-diagonal a_ij (row i, column j:
# the effect of type j on type i, p.7), then the capacity constants.
THETA_NAMES = (
    "r_p",
    "r_tp",
    "r_n",
    "a_p_tp",
    "a_p_n",
    "a_tp_p",
    "a_tp_n",
    "a_n_p",
    "a_n_tp",
)
N_THETA = len(THETA_NAMES)
STATE_NAMES = ("T+", "TP", "T-", "PSA")

# Methods p.7: intrinsic rates from cell-line doubling times, then scaled to 10%
# because "in vivo, these growth rates ... probably represent upper bounds".
R_CELL_LINES = (0.278, 0.355, 0.665)
R_SCALE = 0.1
K_TP_OFF, K_TP_ON = 10_000.0, 100.0
K_N = 10_000.0
CHEAT_OFF, CHEAT_ON = 1.5, 0.5  # T+ cells supported per TP cell
PSA_DECAY = 0.5  # per unit time (Eq. 2)

# Table 3 p.8, (a_p_tp, a_p_n, a_tp_p, a_tp_n, a_n_p, a_n_tp).
PATIENT_1 = (0.7, 0.8, 0.4, 0.5, 0.6, 0.9)
PATIENT_2 = (0.6, 0.8, 0.4, 0.7, 0.5, 0.9)


def theta_for(alphas) -> jnp.ndarray:
    return jnp.array([r * R_SCALE for r in R_CELL_LINES] + list(alphas))


def vector_field(t, y, args):
    """dy/dt for the Zhang 2017 model. `args = (theta, drug)`, drug in [0, 1].

    The paper's drug is on/off; a fractional dose interpolates the two
    capacities linearly, which only a policy with a half dose ever uses.
    """
    theta, drug = args
    r_p, r_tp, r_n, a_p_tp, a_p_n, a_tp_p, a_tp_n, a_n_p, a_n_tp = (
        theta[i] for i in range(N_THETA)
    )
    u = jnp.clip(drug, 0.0, 1.0)
    p, tp, n, psa = (jnp.maximum(y[i], 0.0) for i in range(4))

    k_p = (CHEAT_OFF + (CHEAT_ON - CHEAT_OFF) * u) * tp
    k_tp = K_TP_OFF + (K_TP_ON - K_TP_OFF) * u
    # Guard the division, never shift K: K_T+ is 0 when TP is extinct, and then
    # T+ has no support at all (crowding -> -inf is the model's statement).
    dp = r_p * p * (1.0 - (p + a_p_tp * tp + a_p_n * n) / jnp.maximum(k_p, 1e-10))
    dtp = r_tp * tp * (1.0 - (a_tp_p * p + tp + a_tp_n * n) / k_tp)
    dn = r_n * n * (1.0 - (a_n_p * p + a_n_tp * tp + n) / K_N)
    dpsa = p + tp + n - PSA_DECAY * psa
    return jnp.stack([dp, dtp, dn, dpsa])
