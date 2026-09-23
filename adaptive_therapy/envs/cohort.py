"""Batched virtual-patient rollout.

    a virtual patient = (model_id, theta)
    a cohort          = arrays thereof, vmapped over theta

THE TRAP (PROJECT.md 12, and the reason this module exists rather than a
one-line vmap at each call site): `vmap` over an ADAPTIVE-step solver does not
give each patient its own step count. Under XLA/SIMD the whole batch runs as
many steps as the STIFFEST patient, so a handful of near-singular virtual
patients make every patient pay worst-case and "runs in 30 s" silently becomes
false.

Mitigation, per the spec: fixed-step (`Tsit5` + `ConstantStepSize`) for the
bulk of vmapped rollouts, and `Kvaerno5` with adaptive stepping reserved for a
separately-bucketed stiff minority. `SolverConfig.stiff` selects which.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

from adaptive_therapy.envs import hahnfeldt, kuznetsov, lotka_volterra
from adaptive_therapy.envs.base import ModelId, SolverConfig

_FAMILIES = {
    "lotka_volterra": lotka_volterra,
    "hahnfeldt": hahnfeldt,
    "kuznetsov": kuznetsov,
}


def family(model_id: ModelId):
    """The module implementing `model_id`."""
    try:
        return _FAMILIES[model_id]
    except KeyError:
        raise ValueError(f"unknown model_id: {model_id}") from None


def solve_one_patient(
    model_id: ModelId,
    theta: jnp.ndarray,
    y0: jnp.ndarray,
    dose: float = 0.0,
    cfg: SolverConfig | None = None,
):
    """Integrate one virtual patient under a constant dose.

    A constant dose is the untreated/fixed-schedule case used by the Phase-1
    sanity gate. Closed-loop control replaces `dose` with a policy call between
    decision points; the solver contract does not change.
    """
    cfg = cfg or SolverConfig()
    mod = family(model_id)

    term = diffrax.ODETerm(mod.vector_field)
    ts = jnp.linspace(cfg.t0, cfg.t1, cfg.n_save)

    if cfg.stiff:
        # The stiff bucket only, per PROJECT.md 12. Caution, measured
        # 2026-09-23 (RESEARCH_LOG L166): on Hahnfeldt, Kvaerno5+PID at these
        # tolerances hits max_steps on EVERY patient, while adaptive Tsit5
        # solves all of them. Do not route a real cohort here without
        # re-measuring on that cohort's family.
        solver = diffrax.Kvaerno5()
        controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
    else:
        solver = diffrax.Tsit5()
        controller = diffrax.ConstantStepSize()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=cfg.t0,
        t1=cfg.t1,
        dt0=cfg.dt0,
        y0=y0,
        args=(theta, dose),
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=cfg.max_steps,
    )
    return sol.ys


def solve_cohort(
    model_id: ModelId,
    thetas: jnp.ndarray,
    y0s: jnp.ndarray,
    doses: jnp.ndarray | None = None,
    cfg: SolverConfig | None = None,
):
    """Integrate a cohort. `thetas` (n, n_theta), `y0s` (n, state_dim).

    Returns `(n, n_save, state_dim)`. `model_id` and `cfg` are static, so the
    vmap is over the per-patient axes only -- matching PROJECT.md 6.4's
    `in_axes=(None, 0, 0)`.
    """
    cfg = cfg or SolverConfig()
    n = thetas.shape[0]
    doses = jnp.zeros(n) if doses is None else doses

    def one(theta, y0, dose):
        return solve_one_patient(model_id, theta, y0, dose, cfg)

    return jax.vmap(one)(thetas, y0s, doses)
