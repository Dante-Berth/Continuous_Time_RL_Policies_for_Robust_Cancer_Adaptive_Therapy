"""PREREG M1's pieces: threshold rules, the miss-dose protocol, and cost/turnover bins."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import STROBL_R_S, sample_cohort
from adaptive_therapy.policies.rules import RulePolicy, ThresholdPolicy
from adaptive_therapy.rl.rollout import EpisodeConfig, rollout


def _run(policy, ep: EpisodeConfig):
    c = sample_cohort("lotka_volterra", 1, seed=2, ranges="strobl2021")
    th, y0 = jnp.asarray(c.thetas[0]), jnp.asarray(c.y0s[0])
    return rollout(policy, th, y0, jax.random.PRNGKey(7), ep, RewardConfig(), greedy=True)


def test_threshold_policy_at_0_8_doses_exactly_like_the_thr0_8_rule():
    ep = EpisodeConfig(n_visits=40)
    a = _run(ThresholdPolicy(0.8), ep)["dose"]
    b = _run(RulePolicy("thr0.8"), ep)["dose"]
    assert jnp.array_equal(a, b)


def test_hold_is_the_default_so_a_never_dosing_rule_stays_at_zero():
    traj = _run(RulePolicy("none"), EpisodeConfig(n_visits=40, drop_rate=0.5))
    assert EpisodeConfig().miss_dose == "hold"
    assert float(traj["dose"].max()) == 0.0


def test_full_miss_dose_gives_full_dose_exactly_at_missed_visits():
    traj = _run(RulePolicy("none"), EpisodeConfig(n_visits=40, drop_rate=0.5, miss_dose="full"))
    seen, dose = np.asarray(traj["seen"]), np.asarray(traj["dose"])
    assert (~seen).any()
    assert (dose[~seen] == 1.0).all() and (dose[seen] == 0.0).all()


def test_strobl_bounds_restrict_the_sampled_cost_and_turnover():
    bounds = {"r_R/r_S": (0.5, 0.625), "d_T/r_S": (0.25, 0.5)}
    c = sample_cohort("lotka_volterra", 100, seed=3, ranges="strobl2021", bounds=bounds)
    ratio, turnover = c.thetas[:, 1] / c.thetas[:, 0], c.thetas[:, 3] / STROBL_R_S
    assert (ratio >= 0.5 - 1e-9).all() and (ratio <= 0.625 + 1e-9).all()
    assert (turnover >= 0.25 - 1e-9).all() and (turnover <= 0.5 + 1e-9).all()


def test_bounds_are_refused_outside_strobl2021_or_for_unknown_keys():
    with pytest.raises(ValueError):
        sample_cohort("lotka_volterra", 5, bounds={"r_R/r_S": (0.5, 0.6)})
    with pytest.raises(ValueError):
        sample_cohort("lotka_volterra", 5, ranges="strobl2021", bounds={"r_S": (0.1, 0.2)})
