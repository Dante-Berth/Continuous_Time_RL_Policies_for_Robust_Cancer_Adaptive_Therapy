"""Baez-Kuang two-population cell-quota (Droop) model of androgen suppression.

    dx1/dt = mu(1 - q1/Q) x1 - (D1(Q) + delta1 x1) x1 - lam(Q) x1
    dx2/dt = mu(1 - q2/Q) x2 - (D2(Q) + delta2 x2) x2 + lam(Q) x1
    dQ/dt  = gamma (Qm - Q) - mu [(Q - q1) x1 + (Q - q2) x2] / (x1 + x2)
    dP/dt  = b Q + sigma Q (x1 + x2) - eps P
    Di(Q) = di Ri / (Q + Ri),  lam(Q) = c K / (Q + K),  gamma = gamma1 (1 - u) + gamma2

Growth is set by the androgen quota Q, not by competition for a capacity, and
resistance (x2, q2 < q1) arises by an irreversible, androgen-deprivation-driven
transformation of x1. No pre-existing resistant clone is needed. That is why
this family is the program's second resistance mechanism (ROADMAP V6).

`u` is the dose in [0, 1], 1 = androgen suppression on. The paper's indicator
is the reverse (u = 1 off treatment, eq. 2.6), so it is flipped here so that
every family shares "1 = full dose".

Equations: Phan, He, Martinez & Kuang, MBE 2019;16(1):187-204, eqs. 2.1-2.6
p.189; ranges Table 1 p.192 (read 1 deg, literature/papers/doi-10.3934-mbe.2019010.md).
"""

from __future__ import annotations

import jax.numpy as jnp

# theta layout, fixed. Index order is part of the interface.
THETA_NAMES = (
    "mu", "q1", "q2", "b", "sigma", "eps",
    "d1", "d2", "delta1", "delta2", "R1", "R2",
    "c", "K", "gamma2", "Qm",
)  # fmt: skip
N_THETA = len(THETA_NAMES)
STATE_NAMES = ("x1", "x2", "Q", "P")
GAMMA1 = 20.0  # testes androgen production, fixed in Table 1 p.192

# Table 1 p.192, restricted to the two-population parameters. The paper's lower
# bounds of 0 on R1, R2 and K are lifted to 1e-3: at 0 the Hill terms are
# undefined at Q = 0 and the fit's log-uniform restarts cannot start there.
LO = (0.001, 0.1, 0.1, 0.1e-3, 0.001, 0.001,
      0.001, 0.0001, 0.1e-5, 0.01e-4, 1e-3, 1e-3,
      1e-5, 1e-3, 0.001, 15.0)  # fmt: skip
HI = (0.09, 0.7, 0.5, 2.5e-3, 0.9, 0.01,
      0.09, 0.001, 9e-5, 4.5e-4, 3.0, 3.0,
      1e-4, 1.0, 0.01, 30.0)  # fmt: skip


def vector_field(t, y, args):
    """dy/dt. `args = (theta, dose)`, dose in [0, 1] (1 = suppression on)."""
    theta, dose = args
    mu, q1, q2, b, sigma, eps, d1, d2, de1, de2, R1, R2, c, K, g2, Qm = (
        theta[i] for i in range(N_THETA)
    )
    x1, x2 = jnp.maximum(y[0], 0.0), jnp.maximum(y[1], 0.0)
    Q, P = jnp.maximum(y[2], 1e-10), y[3]
    u = jnp.clip(dose, 0.0, 1.0)

    lam = c * K / (Q + K)
    D1, D2 = d1 * R1 / (Q + R1), d2 * R2 / (Q + R2)
    dx1 = mu * (1.0 - q1 / Q) * x1 - (D1 + de1 * x1) * x1 - lam * x1
    dx2 = mu * (1.0 - q2 / Q) * x2 - (D2 + de2 * x2) * x2 + lam * x1
    gamma = GAMMA1 * (1.0 - u) + g2
    x = x1 + x2
    dQ = gamma * (Qm - Q) - mu * ((Q - q1) * x1 + (Q - q2) * x2) / jnp.maximum(x, 1e-10)
    dP = b * Q + sigma * Q * x - eps * P
    return jnp.stack([dx1, dx2, dQ, dP])
