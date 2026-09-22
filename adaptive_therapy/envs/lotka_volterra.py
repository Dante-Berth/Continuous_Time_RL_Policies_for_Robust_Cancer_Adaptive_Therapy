"""Lotka-Volterra competition (the Gallagher/Strobl baseline).

    dS/dt = r_S*S*(1 - (S+R)/K)*(1 - d_D*u(t)) - d_S*S
    dR/dt = r_R*R*(1 - (S+R)/K)               - d_R*R

Sensitive `S` and resistant `R` compete for one carrying capacity `K`. Cost of
resistance is encoded as r_R < r_S. Treatment `u(t)` in [0,1] suppresses only
the sensitive population's proliferation -- that asymmetry is the whole
mechanism adaptive therapy exploits.

Equations: PROJECT.md 6.1. Parameter ranges: Strobl et al. Cancer Research
2021 (pmc-8455086) -- NOT hardcoded from memory; see literature/papers/.

PROJECT.md 6.1 names the four factors Strobl proved matter, which the cohort
sampler must randomise: initial resistant fraction, proximity to carrying
capacity, resistant proliferation rate, and turnover.
"""

from __future__ import annotations

import jax.numpy as jnp

# theta layout, fixed. Index order is part of the interface: the cohort sampler
# and every test depend on it.
THETA_NAMES = ("r_S", "r_R", "K", "d_S", "d_R", "d_D")
N_THETA = len(THETA_NAMES)
STATE_NAMES = ("S", "R")


def vector_field(t, y, args):
    """dy/dt for the LV competition model. `args = (theta, dose)`."""
    theta, dose = args
    r_S, r_R, K, d_S, d_R, d_D = (theta[i] for i in range(N_THETA))

    S, R = y[0], y[1]
    total = S + R
    # Guard the division rather than adding an epsilon to K: K is a modelled
    # quantity and shifting it changes the equilibrium, which an added epsilon
    # would do silently.
    crowding = 1.0 - total / jnp.maximum(K, 1e-10)

    u = jnp.clip(dose, 0.0, 1.0)
    dS = r_S * S * crowding * (1.0 - d_D * u) - d_S * S
    dR = r_R * R * crowding - d_R * R
    return jnp.stack([dS, dR])


def initial_state(theta: jnp.ndarray, f_R: float, n_0: float) -> jnp.ndarray:
    """Initial (S, R) from resistant fraction `f_R` and capacity fraction `n_0`.

    Parameterised this way because these are two of the four factors Strobl
    showed matter, so the sampler varies them directly rather than varying S
    and R and hoping the ratio lands somewhere interesting.
    """
    K = theta[2]
    total = n_0 * K
    return jnp.stack([total * (1.0 - f_R), total * f_R])
