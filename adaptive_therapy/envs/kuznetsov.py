"""Kuznetsov-Taylor immune predation (Galach 2003 reduction).

    dE/dt = s + p*E*T/(g+T) - m*E*T - dd*E
    dT/dt = a*T*(1 - b*T)    - n*E*T

Effector cells `E` predate tumour `T`, and `T` in turn stimulates effector
recruitment through a saturating term. Produces immunostimulation,
"sneaking through", and a DORMANT state -- dynamics neither LV nor Hahnfeldt
can generate, which is why this family is in the curriculum.

WARNING -- Hopf bifurcations (PROJECT.md 6.3). For some theta the UNTREATED
tumour oscillates rather than growing monotonically. Two consequences that must
be handled before any patient is generated:

  1. The untreated sanity gate must NOT reject oscillatory patients. They are
     the interesting ones -- a policy that only works on monotone growth has
     not been tested on the hard case.
  2. "Crosses the progression threshold once = progression" is WRONG for them.
     `eval/metrics.py` must define the non-monotone TTP rule explicitly.

Equations: PROJECT.md 6.3 / Kuznetsov et al. Bull Math Biol 1994
(doi-10.1007-BF02460644). Non-dimensional values reused in the delay
literature (doi-10.1155-2017-1642976). Both `2` -- unread.
"""

from __future__ import annotations

import jax.numpy as jnp

THETA_NAMES = ("s", "p", "g", "m", "dd", "a", "b", "n")
N_THETA = len(THETA_NAMES)
STATE_NAMES = ("E", "T")

_FLOOR = 0.0


def vector_field(t, y, args):
    """dy/dt for the Kuznetsov-Taylor model. `args = (theta, dose)`.

    Treatment acts on tumour kill directly (a cytotoxic term), not on effector
    recruitment -- modelling an immunotherapy would be a different paper.
    """
    theta, dose = args
    s, p, g, m, dd, a, b, n = (theta[i] for i in range(N_THETA))

    E = jnp.maximum(y[0], _FLOOR)
    T = jnp.maximum(y[1], _FLOOR)

    u = jnp.clip(dose, 0.0, 1.0)

    recruitment = p * E * T / jnp.maximum(g + T, 1e-10)
    dE = s + recruitment - m * E * T - dd * E
    dT = a * T * (1.0 - b * T) - n * E * T - u * T
    return jnp.stack([dE, dT])


def initial_state(theta: jnp.ndarray, t_0: float) -> jnp.ndarray:
    """Initial (E, T) at tumour burden `t_0`, effectors at their tumour-free rest.

    E rest = s/dd, the steady state of dE/dt with T=0. Starting effectors at
    their equilibrium isolates the tumour's effect on the immune system from an
    arbitrary initial transient.
    """
    s, dd = theta[0], theta[4]
    e_rest = s / jnp.maximum(dd, 1e-10)
    return jnp.stack([e_rest, jnp.asarray(t_0)])
