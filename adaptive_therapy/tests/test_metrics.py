"""Invariants of the endpoints.

The TTP tests encode the PROJECT.md 6.3 decision directly: a responder who
relapses must not be scored the same as a patient who never responded.
"""

import numpy as np
import pytest

from adaptive_therapy.eval.metrics import (
    burden_integral,
    cumulative_exposure,
    paired_wilcoxon,
    summarise,
    ttp_from_baseline,
    ttp_running_nadir,
)

TS = np.linspace(0.0, 365.0, 366)


def test_monotone_grower_progresses_when_it_passes_the_threshold():
    """The easy case must still work: steady growth progresses early."""
    b = 100.0 * np.exp(0.01 * TS)[None, :]
    r = ttp_running_nadir(b, TS)
    assert not r.censored[0]
    # +20% over a nadir that never falls => log(1.2)/0.01 ~= 18 d.
    assert 15.0 <= r.days[0] <= 25.0


def test_a_patient_who_never_grows_is_censored_not_scored_at_the_horizon():
    """Censoring is the most common way a TTP comparison lies."""
    b = np.full((1, TS.size), 100.0)
    r = ttp_running_nadir(b, TS)
    assert r.censored[0]
    assert r.n_censored == 1
    assert np.isnan(r.median_iqr()[0])  # nothing observed -> no median


def test_responder_who_relapses_is_credited_for_the_response():
    """THE rule (PROJECT.md 6.3). Burden falls 100x, then regrows.

    Measured from the running nadir, progression is timed from the regrowth,
    not from the start -- otherwise a policy that induces a deep remission and
    then loses control scores identically to one that never worked at all.
    """
    half = TS.size // 2
    down = 100.0 * np.exp(-0.02 * TS[:half])
    up = down[-1] * np.exp(0.05 * (TS[half:] - TS[half - 1]))
    b = np.concatenate([down, up])[None, :]

    r = ttp_running_nadir(b, TS)
    assert not r.censored[0]
    # Progression must be timed after the nadir, not in the first weeks.
    assert r.days[0] > TS[half] - 10.0


def test_nadir_rule_detects_relapse_the_baseline_rule_misses():
    """The case the rule change is FOR (PROJECT.md 6.3).

    A patient who responds deeply and then relapses. The baseline rule cannot
    call progression until burden climbs back above its STARTING value, which
    here takes 91 extra days -- time the policy did not earn. The nadir rule
    times progression from the relapse itself.

    Checked in both directions deliberately: the nadir rule must fire EARLIER
    here. A rule that only ever delays progression would flatter every policy.
    """
    half = 183
    down = 1e5 * np.exp(-0.03 * TS[:half])
    up = down[-1] * np.exp(0.06 * (TS[half:] - TS[half - 1]))
    b = np.concatenate([down, up])[None, :]

    nadir_rule = ttp_running_nadir(b, TS)
    baseline_rule = ttp_from_baseline(b, TS)

    assert not nadir_rule.censored[0]
    assert not baseline_rule.censored[0]
    # Progression is timed at the relapse (~day 183), not at the recrossing.
    assert 180.0 <= nadir_rule.days[0] <= 200.0
    assert baseline_rule.days[0] > nadir_rule.days[0] + 60.0


def test_progression_before_any_response_is_scored_at_that_progression():
    """The other direction: a tumour that grows first HAS progressed.

    The running nadir must not be mistaken for a licence to ignore early
    growth -- a patient who progresses on day 10 and only responds afterwards
    is scored at day 10 by both rules. TTP is time to FIRST progression.
    """
    rise = 1e5 * np.exp(0.02 * TS[:120])
    fall = rise[-1] * np.exp(-0.15 * (TS[120:] - TS[119]))
    b = np.concatenate([rise, fall])[None, :]

    nadir_rule = ttp_running_nadir(b, TS)
    baseline_rule = ttp_from_baseline(b, TS)

    assert not nadir_rule.censored[0]
    assert nadir_rule.days[0] < 30.0
    assert nadir_rule.days[0] == pytest.approx(baseline_rule.days[0])


def test_confirmation_window_rejects_a_transient_spike():
    """A single noisy sample must not end an episode."""
    b = np.full((1, TS.size), 100.0)
    b[0, 50] = 1000.0  # one-sample spike
    r = ttp_running_nadir(b, TS, confirm_days=28.0)
    assert r.censored[0]


def test_confirmation_window_is_in_days_not_samples():
    """Halving the sampling rate must not change the rule's meaning.

    It will change: Tier-4 drops 20-50% of visits, and a window defined in
    samples would silently become twice as strict.
    """
    dense_ts = np.linspace(0.0, 365.0, 366)
    sparse_ts = np.linspace(0.0, 365.0, 53)
    growth = lambda t: 100.0 * np.exp(0.01 * t)  # noqa: E731

    d = ttp_running_nadir(growth(dense_ts)[None, :], dense_ts)
    s = ttp_running_nadir(growth(sparse_ts)[None, :], sparse_ts)
    assert abs(d.days[0] - s.days[0]) < 15.0  # within one sparse interval


@pytest.mark.parametrize("rule", [ttp_running_nadir, ttp_from_baseline])
def test_a_diverged_trajectory_is_refused_never_scored_as_censored(rule):
    ts = np.arange(10) * 7.0
    b = np.vstack([np.linspace(1.0, 2.0, 10), np.full(10, np.nan)])
    with pytest.raises(ValueError, match="non-finite"):
        rule(b, ts)


def test_median_iqr_ignores_censored_patients():
    b = np.stack([100.0 * np.exp(0.01 * TS), np.full(TS.size, 100.0)])
    r = ttp_running_nadir(b, TS)
    assert r.n_censored == 1
    med, q1, q3 = r.median_iqr()
    assert med == pytest.approx(r.days[0], rel=1e-9)  # only the observed one
    assert q1 <= med <= q3


def test_exposure_integrates_dose_over_time():
    doses = np.full((1, TS.size), 0.5)
    assert cumulative_exposure(doses, TS)[0] == pytest.approx(0.5 * 365.0, rel=1e-6)


def test_constant_dosing_costs_more_exposure_than_intermittent():
    """The comparison exposure exists to make fair."""
    always = np.ones((1, TS.size))
    half = np.zeros((1, TS.size))
    half[0, ::2] = 1.0
    assert cumulative_exposure(always, TS)[0] > cumulative_exposure(half, TS)[0]


def test_burden_integral_is_larger_for_a_larger_tumour():
    small = np.full((1, TS.size), 100.0)
    big = np.full((1, TS.size), 1000.0)
    assert burden_integral(big, TS)[0] > burden_integral(small, TS)[0]


def test_paired_wilcoxon_detects_a_consistent_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(100.0, 10.0, size=40)
    b = a - 15.0  # every pair shifted the same way
    res = paired_wilcoxon(a, b)
    assert res["p_value"] < 0.01
    assert res["median_difference"] == pytest.approx(15.0, rel=1e-6)
    assert res["n_pairs"] == 40


def test_paired_wilcoxon_rejects_unmatched_cohorts():
    """Patients are matched by construction; a shape mismatch is a bug."""
    with pytest.raises(ValueError, match="unmatched"):
        paired_wilcoxon(np.zeros(10), np.zeros(11))


def test_identical_arms_are_not_significant():
    a = np.arange(30, dtype=float)
    res = paired_wilcoxon(a, a.copy())
    assert res["p_value"] == 1.0
    assert res["n_tied"] == 30


def test_summarise_reports_every_required_endpoint():
    """PROJECT.md 10: an arm cannot be reported with TTP alone."""
    b = 100.0 * np.exp(0.005 * TS)[None, :]
    d = np.full((1, TS.size), 0.5)
    out = summarise(b, d, TS)
    for key in (
        "ttp_median",
        "ttp_iqr",
        "ttp_censoring_rate",
        "cumulative_exposure_median",
        "burden_integral_median",
        "ttp_secondary_from_baseline",
    ):
        assert key in out, f"missing required endpoint: {key}"
    assert out["ttp_rule"] == "running_nadir"


def test_a_one_sided_wilcoxon_rejects_only_in_the_stated_direction():
    rng = np.random.default_rng(0)
    b = rng.normal(100.0, 10.0, 60)
    worse = b - 5.0 - rng.normal(0.0, 1.0, 60)
    assert paired_wilcoxon(worse, b)["p_value"] < 0.05  # two-sided: "significant"
    assert paired_wilcoxon(worse, b, alternative="greater")["p_value"] > 0.5
    assert paired_wilcoxon(b, worse, alternative="greater")["p_value"] < 0.025


def test_the_capped_median_cannot_reward_letting_more_patients_progress():
    # Arm A: 3 of 4 never progress, one fails at d10. Arm B: all 4 progress,
    # late. The progressor-only median prefers B (it saw A's single early
    # failure); the capped median correctly prefers A.
    from adaptive_therapy.eval.metrics import TTPResult

    h = 357.0
    a = TTPResult(np.array([10.0, h, h, h]), np.array([False, True, True, True]), "t")
    b = TTPResult(np.array([200.0, 210.0, 220.0, 230.0]), np.zeros(4, bool), "t")
    assert b.median_iqr()[0] > a.median_iqr()[0]
    assert a.median_iqr_capped()[0] > b.median_iqr_capped()[0]
