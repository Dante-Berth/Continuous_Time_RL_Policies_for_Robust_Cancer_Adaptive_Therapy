"""Untreated sanity gate and stiffness bucketing.

THE GATE decides which LHS draws become virtual patients. It exists because a
uniform box over theta contains parameter combinations that are numerically
valid but biologically meaningless -- a "tumour" that vanishes untreated, or one
that blows past any plausible burden in a week. Training on those measures the
solver, not the policy.

WHAT THE GATE MUST NOT DO (PROJECT.md 6.3): reject oscillatory patients.
Kuznetsov's Hopf bifurcations make some untreated tumours oscillate or enter
dormancy, and those are the INTERESTING patients -- the ones that break a naive
TTP rule and that a robust policy has to handle. A gate written as "burden must
increase monotonically" would quietly delete the hardest cases and flatter every
policy trained afterwards. The criteria below are therefore about MAGNITUDE and
NUMERICAL VALIDITY, never about shape.

Every rejection is logged with its reason, so the rejection rate and its
composition are reportable rather than invisible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs.base import ModelId, SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_cohort

# A patient must remain within these multiples of its own starting burden to be
# kept. Wide on purpose: this rejects the numerically absurd, not the clinically
# unusual.
MIN_FINAL_FRACTION = 1e-6  # vanished untreated => not a tumour
MAX_GROWTH_FACTOR = 1e6  # unbounded blow-up => solver/parameter pathology

# "Fast-transient" proxy: the largest one-step relative jump in burden.
#
# MEASURED 2026-09-22, and the result revises PROJECT.md 12's prescription.
# The flagged Hahnfeldt patients are NOT stiff in the sense that needs an
# implicit solver -- fixed-step Tsit5 at dt0=0.05 agrees with dt0=0.005 to
# ~1e-4 relative on every one of them. What they have is a fast early
# transient (vasculature equilibrating), which an ADAPTIVE controller thrashes
# on: Kvaerno5 + PIDController exceeds 200k steps and fails on these patients
# even when they are batched ALONE, while fixed-step solves all 24 in 23.5 ms
# with finite results.
#
# So the mitigation is not "bucket the stiff ones off to an implicit solver".
# It is "fixed-step is the default for this simulator, and this flag marks
# patients to re-verify against a finer step rather than to re-solve
# adaptively". The bucketing machinery is kept because a genuinely stiff family
# may yet appear -- but on current evidence the stiff bucket should be EMPTY,
# and a non-empty one is a signal to check accuracy, not to switch solvers.
FAST_TRANSIENT_REL_JUMP = 10.0


@dataclass
class GateReport:
    """Which patients survived, and why the others did not."""

    keep: np.ndarray  # (n,) bool
    reasons: list[str] = field(default_factory=list)  # per rejected patient
    stiff: np.ndarray | None = None  # (n,) bool, among kept

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    @property
    def n_rejected(self) -> int:
        return int((~self.keep).sum())

    @property
    def rejection_rate(self) -> float:
        return float((~self.keep).mean()) if self.keep.size else 0.0

    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(self.reasons))

    def summary(self) -> dict:
        return {
            "n_total": int(self.keep.size),
            "n_kept": self.n_kept,
            "n_rejected": self.n_rejected,
            "rejection_rate": self.rejection_rate,
            "reasons": self.reason_counts(),
            "n_stiff": int(self.stiff.sum()) if self.stiff is not None else 0,
        }


def untreated_sanity_gate(
    model_id: ModelId,
    thetas: np.ndarray,
    y0s: np.ndarray,
    cfg: SolverConfig | None = None,
) -> GateReport:
    """Run each draw untreated and decide whether it is a usable patient."""
    cfg = cfg or SolverConfig()
    ys = solve_cohort(model_id, jnp.asarray(thetas), jnp.asarray(y0s), cfg=cfg)
    burden = np.asarray(observe(ys, model_id))

    n = burden.shape[0]
    keep = np.ones(n, dtype=bool)
    reasons: list[str] = []

    for i in range(n):
        b = burden[i]
        if not np.isfinite(b).all():
            keep[i] = False
            reasons.append("non_finite")
            continue
        if (b < 0).any():
            keep[i] = False
            reasons.append("negative_burden")
            continue
        b0 = b[0]
        if b0 <= 0:
            keep[i] = False
            reasons.append("non_positive_start")
            continue
        # Magnitude only -- never shape. See the module docstring.
        if b.max() > b0 * MAX_GROWTH_FACTOR:
            keep[i] = False
            reasons.append("unbounded_growth")
            continue
        if b[-1] < b0 * MIN_FINAL_FRACTION and b.max() < b0 * 1.01:
            # Vanished without ever growing: not a tumour to treat. A patient
            # that grows first and THEN collapses is dormancy, which is kept.
            keep[i] = False
            reasons.append("vanished_untreated")
            continue

    stiff = np.zeros(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in np.flatnonzero(keep):
            b = np.maximum(burden[i], 1e-30)
            jumps = np.abs(np.diff(np.log(b)))
            stiff[i] = bool(jumps.max() > np.log(FAST_TRANSIENT_REL_JUMP))

    return GateReport(keep=keep, reasons=reasons, stiff=stiff)


def bucket_by_stiffness(
    thetas: np.ndarray, y0s: np.ndarray, report: GateReport
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split kept patients into `bulk` and `fast-transient` cohorts.

    PROJECT.md 12 prescribes sending the second bucket to an implicit solver.
    On measurement that is WRONG for this simulator (see the constant above):
    the flagged patients are accurate under fixed-step and are the ones that
    make an adaptive controller fail. Use this split to RE-VERIFY the flagged
    patients against a finer fixed step, not to switch them to Kvaerno5.

    Kept as a seam because a genuinely stiff family may appear later; the key
    is still named `stiff` so call sites do not have to change if it does.
    """
    kept = np.flatnonzero(report.keep)
    stiff_mask = report.stiff[kept] if report.stiff is not None else np.zeros(kept.size, dtype=bool)
    bulk_idx = kept[~stiff_mask]
    stiff_idx = kept[stiff_mask]
    return {
        "bulk": (thetas[bulk_idx], y0s[bulk_idx]),
        "stiff": (thetas[stiff_idx], y0s[stiff_idx]),
    }
