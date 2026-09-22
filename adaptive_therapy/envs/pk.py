"""Minimal single-compartment pharmacokinetics.

    dC/dt = -clearance * C,  C <- C + bioavailability * dose  at each bolus

PROJECT.md 7 keeps this KNOWN and white-box on purpose: the research question
is whether the POLICY architecture generalises, not whether a network can learn
pharmacokinetics. Making the PK learnable would confound the two.

Drugs are boluses, not infusions (PROJECT.md 7) -- a decision at a monitoring
visit adds a discrete amount to the compartment, which then decays. The policy
therefore emits a DISCRETE dose and the continuous-time part is the state, not
the control.
"""

from __future__ import annotations

import jax.numpy as jnp

from adaptive_therapy.envs.base import PKConfig


def decay(conc: jnp.ndarray, dt: float, cfg: PKConfig) -> jnp.ndarray:
    """Exponential clearance over `dt` days. Closed form, not integrated.

    Solving this analytically rather than adding it to the ODE state keeps the
    stiffness of the tumour system independent of the PK timescale -- a fast
    clearance would otherwise force small steps on the whole system, which is
    the PROJECT.md 12 stiffness trap arriving through the back door.
    """
    return conc * jnp.exp(-cfg.clearance * dt)


def administer(conc: jnp.ndarray, dose: jnp.ndarray, cfg: PKConfig) -> jnp.ndarray:
    """Add a bolus to the compartment."""
    return conc + cfg.bioavailability * dose


def effect(conc: jnp.ndarray, e_max: float = 1.0, ec50: float = 0.5) -> jnp.ndarray:
    """Concentration -> effective treatment intensity u in [0, e_max].

    Saturating (Emax/Hill with n=1) rather than linear: doubling a dose does
    not double cell kill, and a linear map would let a policy buy unbounded
    effect with unbounded exposure, making the exposure penalty the only thing
    standing between the agent and constant maximum dosing.
    """
    return e_max * conc / jnp.maximum(ec50 + conc, 1e-10)
