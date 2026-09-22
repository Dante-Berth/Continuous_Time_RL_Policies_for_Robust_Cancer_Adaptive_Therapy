"""Invariants of the sanity gate, the PK model and the reward.

The gate tests exist mostly to stop one specific regression: a future "tidy-up"
adding a monotone-growth criterion, which would silently delete every
oscillatory patient and flatter every policy trained afterwards.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs import kuznetsov
from adaptive_therapy.envs.base import PKConfig, SolverConfig
from adaptive_therapy.envs.gate import (
    bucket_by_stiffness,
    untreated_sanity_gate,
)
from adaptive_therapy.envs.pk import administer, decay, effect
from adaptive_therapy.envs.reward import RewardConfig, step_reward
from adaptive_therapy.envs.sampling import sample_cohort

CFG = SolverConfig(t1=365.0, n_save=52)


def test_gate_keeps_plausible_patients_across_all_families():
    for fam in ("lotka_volterra", "hahnfeldt", "kuznetsov"):
        c = sample_cohort(fam, 40, seed=0)
        r = untreated_sanity_gate(fam, c.thetas, c.y0s, CFG)
        assert r.n_kept > 0, f"{fam}: gate rejected everything"
        assert r.rejection_rate < 0.5, f"{fam}: {r.rejection_rate:.0%} rejected"


def test_gate_does_not_reject_oscillatory_patients():
    """PROJECT.md 6.3 -- THE criterion the gate must never acquire.

    The Kuznetsov theta that oscillates into dormancy and back must survive.
    These are the interesting patients; a gate that drops them is measuring an
    easier problem than the one being claimed.
    """
    theta = np.array([0.1181, 0.1245, 2.019e7, 3.422e-10, 0.00412, 0.018, 2.0e-9, 1.101e-7])
    y0 = np.asarray(kuznetsov.initial_state(jnp.asarray(theta), t_0=1e5))
    long_cfg = SolverConfig(t1=1200.0, n_save=121)

    r = untreated_sanity_gate("kuznetsov", theta[None], y0[None], long_cfg)
    assert r.keep[0], f"oscillatory patient rejected: {r.reason_counts()}"


def test_gate_rejects_a_tumour_that_vanishes_without_ever_growing():
    """A draw that decays to nothing untreated is not a tumour to treat."""
    theta = np.array([0.001, 0.0005, 1e4, 5.0, 5.0, 1.0])  # huge death rates
    y0 = np.array([100.0, 1.0])
    r = untreated_sanity_gate("lotka_volterra", theta[None], y0[None], CFG)
    assert not r.keep[0]
    assert "vanished_untreated" in r.reason_counts()


def test_gate_reports_every_rejection_with_a_reason():
    """Rejection composition must be reportable, not invisible."""
    theta = np.stack(
        [
            np.array([0.027, 0.008, 1e4, 0.0, 0.0, 1.5]),  # fine
            np.array([0.001, 0.0005, 1e4, 5.0, 5.0, 1.0]),  # vanishes
        ]
    )
    y0 = np.stack([np.array([5000.0, 50.0]), np.array([100.0, 1.0])])
    r = untreated_sanity_gate("lotka_volterra", theta, y0, CFG)
    assert len(r.reasons) == r.n_rejected
    assert set(r.summary()) >= {"n_kept", "n_rejected", "rejection_rate", "reasons"}


def test_bucketing_partitions_the_kept_patients_exactly():
    """No patient may be lost or duplicated by the split."""
    c = sample_cohort("hahnfeldt", 40, seed=0)
    r = untreated_sanity_gate("hahnfeldt", c.thetas, c.y0s, CFG)
    b = bucket_by_stiffness(c.thetas, c.y0s, r)
    assert len(b["bulk"][0]) + len(b["stiff"][0]) == r.n_kept


def test_fixed_step_agrees_with_a_ten_times_finer_step():
    """Fixed-step is the default solver; this is what justifies it.

    Measured 2026-09-22: on the patients the fast-transient flag marks,
    dt0=0.05 and dt0=0.005 agree to ~1e-4 relative. If a future change breaks
    that, fixed-step stops being safe and this fails.
    """
    from adaptive_therapy.envs.base import observe
    from adaptive_therapy.envs.cohort import solve_one_patient

    c = sample_cohort("hahnfeldt", 8, seed=0)
    for i in range(len(c.thetas)):
        coarse = observe(
            solve_one_patient(
                "hahnfeldt",
                jnp.asarray(c.thetas[i]),
                jnp.asarray(c.y0s[i]),
                0.0,
                SolverConfig(t1=365.0, n_save=52, dt0=0.05),
            ),
            "hahnfeldt",
        )
        fine = observe(
            solve_one_patient(
                "hahnfeldt",
                jnp.asarray(c.thetas[i]),
                jnp.asarray(c.y0s[i]),
                0.0,
                SolverConfig(t1=365.0, n_save=52, dt0=0.005),
            ),
            "hahnfeldt",
        )
        rel = float(jnp.max(jnp.abs(coarse - fine) / jnp.maximum(jnp.abs(fine), 1e-12)))
        assert rel < 1e-3, f"patient {i}: fixed-step diverges, rel={rel:.2e}"


# --- PK ---------------------------------------------------------------------


def test_concentration_decays_exponentially():
    cfg = PKConfig(clearance=0.5)
    c0 = jnp.array(1.0)
    assert float(decay(c0, 1.0, cfg)) == pytest.approx(float(jnp.exp(-0.5)), rel=1e-6)


def test_a_bolus_raises_concentration_then_clears():
    cfg = PKConfig()
    c = administer(jnp.array(0.0), jnp.array(1.0), cfg)
    assert float(c) == pytest.approx(1.0, rel=1e-6)
    assert float(decay(c, 10.0, cfg)) < 0.05


def test_drug_effect_saturates_rather_than_scaling_linearly():
    """A linear map would let a policy buy unbounded effect with dose."""
    assert float(effect(jnp.array(1e6))) == pytest.approx(1.0, rel=1e-3)
    low, high = float(effect(jnp.array(0.5))), float(effect(jnp.array(1.0)))
    assert high < 2 * low  # doubling dose does not double effect


# --- reward -----------------------------------------------------------------


def test_reward_pays_for_survival_and_charges_for_dose():
    cfg = RewardConfig(exposure_penalty=0.1)
    b = n = jnp.array(100.0)
    undosed = float(step_reward(b, n, jnp.array(0.0), 1.0, cfg))
    dosed = float(step_reward(b, n, jnp.array(1.0), 1.0, cfg))
    assert undosed == pytest.approx(cfg.survival_bonus, rel=1e-6)
    assert dosed < undosed  # constant dosing is not free


def test_reward_penalises_burden_above_the_nadir_threshold():
    """The reward's progression notion must match the TTP rule's."""
    cfg = RewardConfig()
    nadir = jnp.array(100.0)
    at = float(step_reward(jnp.array(110.0), nadir, jnp.array(0.0), 1.0, cfg))
    over = float(step_reward(jnp.array(200.0), nadir, jnp.array(0.0), 1.0, cfg))
    assert at == pytest.approx(cfg.survival_bonus, rel=1e-6)  # below +20%
    assert over < at
