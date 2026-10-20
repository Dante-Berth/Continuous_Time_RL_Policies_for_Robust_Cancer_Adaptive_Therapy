"""de Pillis-Gu-Radunskaya tumour / NK / CD8+ T / lymphocyte model with chemotherapy.

    dT/dt = a T (1 - b T) - c N T - D T - K_T (1 - e^-M) T
    dN/dt = e C - f N + g T^2/(h + T^2) N - p N T - K_N (1 - e^-M) N
    dL/dt = -m L + j D^2 T^2/(k + D^2 T^2) L - q L T + (r1 N + r2 C) T
            - u N L^2 - K_L (1 - e^-M) L + p_I L I/(g_I + I) + v_L(t)
    dC/dt = alpha - beta C - K_C (1 - e^-M) C
    dM/dt = -gamma M + v_M(t)
    dI/dt = -mu_I I + v_I(t),            D = d (L/T)^l / (s + (L/T)^l)

Eqs. 1-8 of de Pillis et al., J Theor Biol 238:841-862, 2006
(doi-10.1016-j.jtbi.2005.06.037, read 1°). The point of this family in the
program (PREREG W13): chemotherapy kills the immune compartments too
(K_N = K_L = K_C < K_T, the same order), so full dosing is not optimal by
construction, unlike Hahnfeldt/Kuznetsov (CLOSED #3). There is NO resistant
compartment; the mechanism is immune escape and bistability.

Parameter values are transcribed from Tables 1-2 of the paper, read off the
rendered pages (the text layer of the PDF drops the exponent signs). The
mouse set has no IL-2 constants; the human values (Kirschner-Panetta) are
used, which is harmless because v_I is zero in every mouse experiment.
Inputs v_M, v_L, v_I are held CONSTANT over a day: the paper doses in 1-day
pulses.
"""

from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp

THETA_NAMES = (
    "a", "b", "c", "d", "l", "s", "e", "f", "g", "h", "p", "m", "j", "k", "q",
    "r1", "r2", "u", "K_T", "K_N", "K_L", "K_C", "alpha", "beta", "gamma",
    "p_I", "g_I", "mu_I",
)  # fmt: skip
N_THETA = len(THETA_NAMES)
STATE_NAMES = ("T", "N", "L", "C", "M", "I")

_IL2 = {"p_I": 1.25e-1, "g_I": 2.0e7, "mu_I": 1.0e1}  # Table 2 (Kirschner & Panetta 1998)
_CHEMO = {"K_T": 9.0e-1, "K_N": 6.0e-1, "K_L": 6.0e-1, "K_C": 6.0e-1, "gamma": 9.0e-1}

# Table 1 (mouse; Diefenbach et al. 2001 and others).
MOUSE = dict(
    a=4.31e-1, b=2.17e-8, c=7.13e-10, d=8.17, l=6.57e-1, s=6.18e-1, e=1.29e-3,
    f=4.12e-2, g=4.98e-1, h=2.02e7, p=1.0e-7, m=2.0e-2, j=9.96e-1, k=3.03e5,
    q=3.42e-10, r1=1.1e-7, r2=3.0e-11, u=1.80e-8, alpha=1.21e5, beta=1.2e-2,
    **_CHEMO, **_IL2,
)  # fmt: skip

# Table 2 (human; Dudley et al. 2002). Patients 9 and 10 differ in d, l, k, m, q, p, s, alpha, beta.
_HUMAN = dict(
    a=4.31e-1, b=1.02e-9, c=6.41e-11, e=2.08e-7, f=4.12e-2, g=1.25e-2, h=2.02e7,
    j=2.49e-2, r1=1.10e-7, r2=6.50e-11, u=3.00e-10, **_CHEMO, **_IL2,
)  # fmt: skip
PATIENT9 = dict(
    _HUMAN, d=2.34, l=2.09, k=3.66e7, m=2.04e-1, q=1.42e-6, p=3.42e-6, s=8.39e-2,
    alpha=7.50e8, beta=1.20e-2,
)  # fmt: skip
PATIENT10 = dict(
    _HUMAN, d=1.88, l=1.81, k=5.66e7, m=9.12, q=1.59e-6, p=3.59e-6, s=5.12e-1,
    alpha=5.00e8, beta=8.00e-3,
)  # fmt: skip

PRESETS = {"mouse": MOUSE, "patient9": PATIENT9, "patient10": PATIENT10}


def theta_from(params: dict) -> jnp.ndarray:
    """Parameter dict -> vector in THETA_NAMES order (KeyError if one is missing)."""
    return jnp.array([params[n] for n in THETA_NAMES])


def vector_field(t, y, args):
    """dy/dt, `args = (theta, (v_M, v_L, v_I))`; inputs are constant over the call."""
    theta, (v_m, v_l, v_i) = args
    (a, b, c, d, l, s, e, f, g, h, p, m, j, k, q, r1, r2, uu, K_T, K_N, K_L, K_C,  # noqa: E741 (the paper symbol l)
     alpha, beta, gamma, p_I, g_I, mu_I) = (theta[i] for i in range(N_THETA))  # fmt: skip

    T = jnp.maximum(y[0], 0.0)
    N = jnp.maximum(y[1], 0.0)
    L = jnp.maximum(y[2], 0.0)
    C = jnp.maximum(y[3], 0.0)
    M = jnp.maximum(y[4], 0.0)
    I = jnp.maximum(y[5], 0.0)  # noqa: E741

    # (L/T)^l with T -> 0 is the paper's own singular point; the guard is
    # jnp.maximum, not an added epsilon, so non-degenerate dynamics are untouched.
    ratio = jnp.power(L / jnp.maximum(T, 1e-10), l)
    D = d * ratio / jnp.maximum(s + ratio, 1e-30)
    kill = 1.0 - jnp.exp(-M)
    D2T2 = (D * T) ** 2

    dT = a * T * (1.0 - b * T) - c * N * T - D * T - K_T * kill * T
    dN = e * C - f * N + g * T**2 / (h + T**2) * N - p * N * T - K_N * kill * N
    dL = (
        -m * L
        + j * D2T2 / (k + D2T2) * L
        - q * L * T
        + (r1 * N + r2 * C) * T
        - uu * N * L**2
        - K_L * kill * L
        + p_I * L * I / (g_I + I)
        + v_l
    )
    dC = alpha - beta * C - K_C * kill * C
    dM = -gamma * M + v_m
    dI = -mu_I * I + v_i
    return jnp.stack([dT, dN, dL, dC, dM, dI])


def simulate_days(theta, y0, v_m, v_l, v_i, rtol: float = 1e-8):
    """Integrate day by day with the inputs of that day held constant.

    `v_m`, `v_l`, `v_i`: arrays of length `n_days`. Returns the state at the
    end of every day, shape (n_days + 1, 6), row 0 = `y0`. Float64 is needed:
    cell counts span ~1e0..1e10 and the rate constants ~1e-11.
    """
    term = dfx.ODETerm(vector_field)
    solver = dfx.Tsit5()
    ctrl = dfx.PIDController(rtol=rtol, atol=1e-6)

    def one_day(y, u):
        sol = dfx.diffeqsolve(
            term, solver, 0.0, 1.0, 1e-3, y, args=(theta, u),
            stepsize_controller=ctrl, max_steps=100_000,
        )  # fmt: skip
        y1 = sol.ys[-1] if sol.ys is not None else sol.y1
        return y1, y1

    _, ys = jax.lax.scan(one_day, y0, (v_m, v_l, v_i))
    return jnp.concatenate([y0[None], ys], axis=0)
