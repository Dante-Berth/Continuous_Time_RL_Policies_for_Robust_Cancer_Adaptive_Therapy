"""Latin-hypercube sampling of virtual-patient cohorts.

A virtual patient is `(model_id, theta)`. This module owns the per-family theta
ranges and the untreated sanity gate that decides which draws are kept.

THETA_RANGES ARE PROVISIONAL. They are order-of-magnitude plausible and
internally consistent, but their literature sources are `2` (unread) in
literature/PAPERS.md, so per REVIEW_PROTOCOL.md 0.3 none of them may be cited
as justification for a design choice. A number computed from them reaches a
paper only as a declared non-literature regime.

The exception is `ranges="strobl2021"` (LV only): every value is Strobl et al.
2021 Table 1, read 1° from the PDF (literature/papers/pmc-8455086.md, p.29,
2026-09-30).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_therapy.envs import hahnfeldt, kuznetsov, lotka_volterra
from adaptive_therapy.envs.base import ModelId

# (low, high) per theta component, sampled log-uniformly where the range spans
# more than one order of magnitude.
THETA_RANGES: dict[ModelId, dict[str, tuple[float, float]]] = {
    "lotka_volterra": {
        "r_S": (0.015, 0.045),
        "r_R": (0.004, 0.014),  # cost of resistance: r_R < r_S by construction
        "K": (5e3, 2e4),
        "d_S": (0.0, 0.01),
        "d_R": (0.0, 0.01),
        "d_D": (0.8, 2.0),
    },
    "hahnfeldt": {
        "a": (0.10, 0.30),
        "b": (3.0, 9.0),
        "d": (0.004, 0.015),
        "e": (0.3, 1.2),
        "K_max": (5e3, 2e4),
    },
    "kuznetsov": {
        "s": (0.05, 0.20),
        "p": (0.08, 0.20),
        "g": (1e7, 4e7),
        "m": (1e-10, 8e-10),
        "dd": (0.002, 0.008),
        "a": (0.010, 0.030),
        "b": (1e-9, 4e-9),
        "n": (5e-8, 2e-7),
    },
}

# Initial-condition ranges. For LV these are two of the four factors Strobl
# showed matter (PROJECT.md 6.1), so they are sampled rather than fixed.
IC_RANGES: dict[ModelId, dict[str, tuple[float, float]]] = {
    "lotka_volterra": {"f_R": (0.001, 0.10), "n_0": (0.30, 0.85)},
    "hahnfeldt": {"v_0": (50.0, 500.0)},
    "kuznetsov": {"t_0": (1e4, 5e5)},
}

# LV ranges from Strobl et al. 2021, Cancer Research, Table 1
# (doi 10.1158/0008-5472.CAN-20-0806; checked 1° against the PDF, p.29, on
# 2026-09-30: every value matches). 2026-09-29 (RESEARCH_LOG L703): only 9.8% of
# "provisional" LV patients fall inside these ranges -- r_R/r_S median 0.30
# against Strobl's 0.5-1.0, i.e. a far stronger cost of resistance. r_S and the
# drug kill are fixed as in the paper; one turnover d_T acts on both
# populations; K only sets the scale (Strobl normalises it away).
STROBL_R_S = 0.027  # /day, Table 1
STROBL_D_D = 1.5  # drug kill, Table 1: (1 - 1.5 u) at full dose u = 1
STROBL_LV = {
    "r_R/r_S": (0.5, 1.0),
    "K": (5e3, 2e4),
    "d_T/r_S": (0.0, 0.5),
    "f_R": (0.001, 0.10),
    "n_0": (0.10, 0.75),
}
RANGE_SETS = ("provisional", "strobl2021")

_FAMILY_MODULES = {
    "lotka_volterra": lotka_volterra,
    "hahnfeldt": hahnfeldt,
    "kuznetsov": kuznetsov,
}


@dataclass(frozen=True)
class Cohort:
    """A sampled cohort, post-gate."""

    model_id: ModelId
    thetas: np.ndarray  # (n, n_theta)
    y0s: np.ndarray  # (n, state_dim)
    n_requested: int
    n_rejected: int

    @property
    def n(self) -> int:
        return self.thetas.shape[0]

    @property
    def rejection_rate(self) -> float:
        return self.n_rejected / max(self.n_requested, 1)


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube on [0,1]^d: one stratified sample per row, per column."""
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, d))
    points = cut[:n, None] + u * (cut[1] - cut[0])
    for j in range(d):
        points[:, j] = rng.permutation(points[:, j])
    return points


def _scale(unit: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map [0,1] to [lo,hi], log-uniformly when the range spans >1 decade."""
    if lo > 0 and hi / lo > 10:
        return np.exp(np.log(lo) + unit * (np.log(hi) - np.log(lo)))
    return lo + unit * (hi - lo)


def _strobl_lv(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(thetas, ics) in the LV layout from a (n, 5) unit LHS over STROBL_LV."""
    v = {k: _scale(unit[:, j], *r) for j, (k, r) in enumerate(STROBL_LV.items())}
    n = unit.shape[0]
    r_S = np.full(n, STROBL_R_S)
    d_T = v["d_T/r_S"] * STROBL_R_S
    thetas = np.column_stack([r_S, v["r_R/r_S"] * r_S, v["K"], d_T, d_T, np.full(n, STROBL_D_D)])
    return thetas, np.column_stack([v["f_R"], v["n_0"]])


def sample_cohort(model_id: ModelId, n: int, seed: int = 0, ranges: str = "provisional") -> Cohort:
    """LHS-sample `n` virtual patients, then apply the untreated sanity gate.

    `ranges` selects the theta box: "provisional" (every run up to W4b) or
    "strobl2021" (LV only). The default stays "provisional" so logged runs
    reproduce bit-for-bit.
    """
    if ranges not in RANGE_SETS:
        raise ValueError(f"unknown ranges {ranges!r}; expected one of {RANGE_SETS}")
    if ranges == "strobl2021" and model_id != "lotka_volterra":
        raise ValueError("strobl2021 ranges exist for lotka_volterra only")
    rng = np.random.default_rng(seed)
    theta_spec = THETA_RANGES[model_id]
    ic_spec = IC_RANGES[model_id]

    n_draw = int(n * 1.6) + 16  # headroom so the gate still leaves ~n
    if ranges == "strobl2021":
        thetas, ics = _strobl_lv(_lhs(n_draw, len(STROBL_LV), rng))
    else:
        unit = _lhs(n_draw, len(theta_spec) + len(ic_spec), rng)
        thetas = np.column_stack(
            [_scale(unit[:, j], *rng_) for j, rng_ in enumerate(theta_spec.values())]
        )
        ics = np.column_stack(
            [_scale(unit[:, len(theta_spec) + j], *rng_) for j, rng_ in enumerate(ic_spec.values())]
        )
    if ranges == "provisional" and model_id == "lotka_volterra":
        # Enforce the cost of resistance rather than rejecting draws that
        # violate it: r_R >= r_S is a different biology, not a bad sample.
        r_S, r_R = thetas[:, 0], thetas[:, 1]
        thetas[:, 1] = np.minimum(r_R, r_S * 0.95)

    mod = _FAMILY_MODULES[model_id]
    y0s = np.stack(
        [np.asarray(mod.initial_state(thetas[i], *ics[i]), dtype=float) for i in range(n_draw)]
    )

    keep = np.isfinite(thetas).all(axis=1) & np.isfinite(y0s).all(axis=1)
    keep &= (y0s > 0).all(axis=1)

    idx = np.flatnonzero(keep)[:n]
    return Cohort(
        model_id=model_id,
        thetas=thetas[idx],
        y0s=y0s[idx],
        n_requested=n_draw,
        n_rejected=int(n_draw - keep.sum()),
    )
