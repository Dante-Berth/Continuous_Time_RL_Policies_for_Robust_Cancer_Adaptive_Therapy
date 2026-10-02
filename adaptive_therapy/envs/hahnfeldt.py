"""Hahnfeldt angiogenesis: tumour volume with a dynamic vascular capacity.

    dV/dt = -a*V*ln(V/K)
    dK/dt =  b*V - d*V^(2/3)*K - e*u(t)*K

The signature assumption is the `V^(2/3)`: inhibition is released through the
tumour SURFACE, so the inhibitory term scales with area rather than volume
(I/S ~ K^theta * x^mu with theta+mu = 2/3).

Why this family earns its place (PROJECT.md 6.2): the bottleneck is
VASCULATURE, not competition for a fixed capacity. An LV-tuned burden threshold
has no mechanistic reason to transfer here, which is what makes the Tier-2
structural shift a real test rather than a re-parametrisation.

Equations: Hahnfeldt et al. Cancer Research 1999, Eqs. C1/C4 (read 1 deg
2026-10-02; lambda2 = 0 as fitted there). Ranges: PLOS One 2015
(doi-10.1371-journal.pone.0120007), still 2 deg. The paper's drug term is
e*K*g(t) with g in mg/kg, not a dose clipped to [0, 1]: through this function,
pass dose = 1 and e*g in theta (research/v5_repro.py). W5-pre's normalised
dose made the drug ~10x too weak (RESEARCH_LOG L1176).
"""

from __future__ import annotations

import jax.numpy as jnp

THETA_NAMES = ("a", "b", "d", "e", "K_max")
N_THETA = len(THETA_NAMES)
STATE_NAMES = ("V", "K")

# Gompertzian growth is singular at V=0 (ln(V/K) -> -inf). Virtual patients
# never start there, but a solver overshoot can, and a NaN at step 3 of a
# 40-minute cohort solve is expensive to diagnose. Floor both state variables.
_FLOOR = 1e-8


def vector_field(t, y, args):
    """dy/dt for the Hahnfeldt model. `args = (theta, dose)`."""
    theta, dose = args
    a, b, d, e, _K_max = (theta[i] for i in range(N_THETA))

    V = jnp.maximum(y[0], _FLOOR)
    K = jnp.maximum(y[1], _FLOOR)

    u = jnp.clip(dose, 0.0, 1.0)

    dV = -a * V * jnp.log(V / K)
    # The surface term. cbrt rather than V**(2/3): the latter is evaluated as
    # exp((2/3)*log(V)) and loses precision at the small volumes that a
    # responding tumour actually reaches.
    surface = jnp.cbrt(V) ** 2
    dK = b * V - d * surface * K - e * u * K
    return jnp.stack([dV, dK])


def initial_state(theta: jnp.ndarray, v_0: float) -> jnp.ndarray:
    """Initial (V, K) at volume `v_0`, vasculature at carrying equilibrium.

    Starting with K = V rather than K = K_max: a tumour at presentation has
    already grown its own vasculature, so starting it vascular-rich would hand
    every policy an easier problem than the biology poses.
    """
    return jnp.stack([jnp.asarray(v_0), jnp.asarray(v_0)])
