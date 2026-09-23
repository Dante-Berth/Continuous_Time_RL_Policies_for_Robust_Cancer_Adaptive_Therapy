"""Endpoints: TTP, cumulative exposure, burden integral, paired statistics.

THE TTP RULE (PROJECT.md 6.3 -- decided 2026-09-22, before any policy exists)

Kuznetsov's Hopf bifurcations make some untreated tumours oscillate: burden
crosses a threshold on the way up, collapses into dormancy, then recovers
months later. A "first crossing = progression" rule calls such a patient
progressed on day 11 while they spend 65% of the following three years below
that threshold, and a nadir-anchored rule with no confirmation calls them
progressed on day 0. Both are wrong, and the difference is not marginal -- it
is the whole trajectory.

The rule used here is RECIST-style and clinically defensible:

    progression at the first t where burden exceeds the RUNNING NADIR (best
    response achieved so far) by a factor (1+delta), AND stays above that
    bound for a confirmation window W.

Measuring from the running nadir credits a patient for a response before
penalising the relapse; the confirmation window stops a single noisy sample or
a transient oscillation peak from ending an episode. `ttp_from_baseline` is
computed alongside as a SECONDARY diagnostic so the sensitivity of any
conclusion to the rule stays visible -- it is never the reported endpoint.

Censoring: a patient who never progresses within the horizon is CENSORED at the
horizon, not assigned the horizon as a value. Treating censored patients as
having progressed on the last day silently understates a good policy, and it is
the most common way a TTP comparison lies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Defaults. PROJECT.md 7 uses a +20% burden rise; the 28 d confirmation is one
# monitoring interval at the weekly sampling used in Phase 0.
DEFAULT_DELTA = 0.20
DEFAULT_CONFIRM_DAYS = 28.0


@dataclass(frozen=True)
class TTPResult:
    """Time to progression for one cohort.

    `days` holds the progression time; `censored` marks patients who never
    progressed within the horizon. Any summary MUST respect the mask.
    """

    days: np.ndarray  # (n,)
    censored: np.ndarray  # (n,) bool
    rule: str

    @property
    def n_censored(self) -> int:
        return int(self.censored.sum())

    @property
    def censoring_rate(self) -> float:
        return float(self.censored.mean())

    def median_iqr(self) -> tuple[float, float, float]:
        """Median and IQR over OBSERVED progressions only.

        Median, never mean: PROJECT.md 10 -- trajectories are heavy-tailed and
        a mean is dominated by whichever patients happened not to progress.
        Returns NaNs when nothing progressed, rather than a misleading number.
        """
        obs = self.days[~self.censored]
        if obs.size == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.median(obs)),
            float(np.percentile(obs, 25)),
            float(np.percentile(obs, 75)),
        )


def _finite_burden(burden: np.ndarray) -> np.ndarray:
    """Refuse a non-finite trajectory rather than score it.

    2026-09-23: NaN never compares above the nadir, so a blown-up rollout was
    silently scored CENSORED -- the best possible outcome. That happened to
    14-18% of Phase 2's first Tier-1 cohort under full dose and corrupted the
    gate. A diverged patient is a simulator or cohort bug, never a TTP.
    """
    burden = np.atleast_2d(np.asarray(burden, dtype=float))
    bad = ~np.isfinite(burden).all(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())}/{bad.size} burden trajectories are non-finite")
    return burden


def _confirmed_crossing(above: np.ndarray, ts: np.ndarray, confirm_days: float) -> int:
    """First index where `above` holds continuously for `confirm_days`.

    Returns -1 if never. The window is in DAYS, not samples, so the rule does
    not silently change meaning when the sampling rate changes -- which it will,
    for the Tier-4 irregular-sampling experiments.
    """
    n = above.size
    for i in range(n):
        if not above[i]:
            continue
        end = ts[i] + confirm_days
        j = i
        ok = True
        while j < n and ts[j] <= end:
            if not above[j]:
                ok = False
                break
            j += 1
        # Reaching the end of the horizon without falsifying the window counts
        # as confirmed: the alternative is that progression can never be
        # detected in the final `confirm_days`, which biases late progressions
        # into the censored bucket.
        if ok:
            return i
    return -1


def ttp_running_nadir(
    burden: np.ndarray,
    ts: np.ndarray,
    delta: float = DEFAULT_DELTA,
    confirm_days: float = DEFAULT_CONFIRM_DAYS,
) -> TTPResult:
    """THE primary endpoint. `burden` (n, T), `ts` (T,) in days."""
    burden = _finite_burden(burden)
    ts = np.asarray(ts, dtype=float)
    n = burden.shape[0]

    days = np.full(n, ts[-1], dtype=float)
    censored = np.ones(n, dtype=bool)

    for k in range(n):
        b = burden[k]
        nadir = np.minimum.accumulate(b)
        above = b > nadir * (1.0 + delta)
        idx = _confirmed_crossing(above, ts, confirm_days)
        if idx >= 0:
            days[k] = ts[idx]
            censored[k] = False

    return TTPResult(days=days, censored=censored, rule="running_nadir")


def ttp_from_baseline(
    burden: np.ndarray,
    ts: np.ndarray,
    delta: float = DEFAULT_DELTA,
    confirm_days: float = DEFAULT_CONFIRM_DAYS,
) -> TTPResult:
    """SECONDARY diagnostic only -- never the reported endpoint.

    Kept so every result can show how sensitive it is to the TTP rule.
    """
    burden = _finite_burden(burden)
    ts = np.asarray(ts, dtype=float)
    n = burden.shape[0]

    days = np.full(n, ts[-1], dtype=float)
    censored = np.ones(n, dtype=bool)

    for k in range(n):
        above = burden[k] > burden[k, 0] * (1.0 + delta)
        idx = _confirmed_crossing(above, ts, confirm_days)
        if idx >= 0:
            days[k] = ts[idx]
            censored[k] = False

    return TTPResult(days=days, censored=censored, rule="from_baseline")


def cumulative_exposure(doses: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Integral of dose over time, per patient. (n, T) -> (n,).

    PROJECT.md 10: report this with EVERY TTP number. A policy that wins TTP by
    dosing constantly is aggressive, not robust, and without this the
    comparison against a binary baseline is not fair.
    """
    doses = np.atleast_2d(np.asarray(doses, dtype=float))
    return np.trapezoid(doses, x=np.asarray(ts, dtype=float), axis=1)


def burden_integral(burden: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Area under the burden curve -- the second endpoint (PROJECT.md 5/10).

    A metastasis proxy: total tumour-cell-days is monotone in the opportunity
    for a seeding event. Reported alongside TTP because the two pull against
    each other -- a TTP-maximising schedule can worsen metastasis risk, and
    reporting TTP alone optimises the wrong thing.
    """
    burden = np.atleast_2d(np.asarray(burden, dtype=float))
    return np.trapezoid(burden, x=np.asarray(ts, dtype=float), axis=1)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon signed-rank across MATCHED virtual patients.

    PROJECT.md 10: not a t-test. Patients are matched by construction (same
    theta, different policy), the differences are not normal, and the absence
    of this test is an instant reviewer flag.

    `alternative="greater"` tests a > b. 2026-09-23: Phase 2 used two-sided p
    alone, which is "significant" whichever arm wins -- a gate asking whether
    ODE beats MLP must test that direction (PREREG Phase 2b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"unmatched cohorts: {a.shape} vs {b.shape}")

    diff = a - b
    n_tied = int((diff == 0).sum())
    if np.all(diff == 0):
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "n_pairs": int(a.size),
            "n_tied": n_tied,
            "median_difference": 0.0,
        }

    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative=alternative)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n_pairs": int(a.size),
        "n_tied": n_tied,
        "median_difference": float(np.median(diff)),
    }


def summarise(
    burden: np.ndarray,
    doses: np.ndarray,
    ts: np.ndarray,
    delta: float = DEFAULT_DELTA,
    confirm_days: float = DEFAULT_CONFIRM_DAYS,
) -> dict:
    """The full PROJECT.md 10 reporting block for one arm.

    Returns every endpoint the spec requires together, so an arm cannot be
    reported with TTP alone.
    """
    primary = ttp_running_nadir(burden, ts, delta, confirm_days)
    secondary = ttp_from_baseline(burden, ts, delta, confirm_days)
    med, q1, q3 = primary.median_iqr()
    smed, sq1, sq3 = secondary.median_iqr()

    return {
        "ttp_rule": primary.rule,
        "ttp_median": med,
        "ttp_iqr": [q1, q3],
        "ttp_censoring_rate": primary.censoring_rate,
        "ttp_n_censored": primary.n_censored,
        "ttp_secondary_from_baseline": {
            "median": smed,
            "iqr": [sq1, sq3],
            "censoring_rate": secondary.censoring_rate,
        },
        "cumulative_exposure_median": float(np.median(cumulative_exposure(doses, ts))),
        "burden_integral_median": float(np.median(burden_integral(burden, ts))),
        "n_patients": int(np.atleast_2d(burden).shape[0]),
    }
