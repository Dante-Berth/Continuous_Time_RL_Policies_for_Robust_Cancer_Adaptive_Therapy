"""W8 invariants: a visit schedule is deterministic, and lookahead tells the truth.

PREREG W8: the policy is told how many slots remain to its next visit. If that
number were off by one, or if a schedule changed what a regular episode does,
the scheduled-visit comparison would measure the bug.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.policies.lstm import LSTMPolicy
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy
from adaptive_therapy.policies.rules import RulePolicy
from adaptive_therapy.rl.rollout import (
    LOOKAHEAD_SCALE,
    EpisodeConfig,
    rollout,
    schedule_mask,
    slots_to_next_visit,
)

KEY = jax.random.PRNGKey(0)


def test_a_schedule_visits_slot_zero_then_follows_its_cyclic_gaps():
    m = schedule_mask(12, (1, 3))
    assert np.flatnonzero(m).tolist() == [0, 1, 4, 5, 8, 9]
    assert schedule_mask(52, (1, 3)).sum() == 26  # the 50%-missed visit count
    assert schedule_mask(52, (2, 2)).sum() == 26
    assert schedule_mask(52, (1, 2, 3)).sum() == 27  # same mean gap; the cycle leaves one extra
    with pytest.raises(ValueError):
        schedule_mask(10, (0, 2))


def test_lookahead_counts_slots_to_the_next_observed_visit():
    m = schedule_mask(10, (1, 3))  # visits at 0, 1, 4, 5, 8, 9
    ahead = slots_to_next_visit(m)
    assert ahead[m].tolist() == [1.0, 3.0, 1.0, 3.0, 1.0, 1.0]  # last: to the episode end
    assert ahead[2] == 2.0 and ahead[3] == 1.0  # unseen slots count down too


def _patient():
    c = sample_cohort("lotka_volterra", 1, seed=0)
    return jnp.asarray(c.thetas[0]), jnp.asarray(c.y0s[0])


def test_a_schedule_of_one_slot_is_the_regular_episode():
    th, y0 = _patient()
    rw, p = RewardConfig(exposure_penalty=0.1), RulePolicy("thr0.8")
    a = rollout(p, th, y0, KEY, EpisodeConfig(), rw, greedy=True)
    b = rollout(p, th, y0, KEY, EpisodeConfig(schedule=(1,)), rw, greedy=True)
    for k in ("burden", "dose", "seen"):
        assert jnp.allclose(a[k], b[k])


def test_the_policy_is_shown_the_true_gap_to_its_next_visit():
    th, y0 = _patient()
    ep = EpisodeConfig(schedule=(1, 3), lookahead=True)
    traj = rollout(MLPPolicy(KEY, width=386, layer_norm=True, in_dim=2), th, y0, KEY, ep, RewardConfig())
    seen = np.asarray(traj["seen"])
    got = np.asarray(traj["obs"])[:, 1] * LOOKAHEAD_SCALE
    assert np.allclose(got, slots_to_next_visit(seen))
    assert traj["obs"].shape == (52, 2)


def test_a_missed_slot_repeats_the_last_dose_under_a_schedule():
    th, y0 = _patient()
    ep = EpisodeConfig(schedule=(1, 3), lookahead=True)
    traj = rollout(MLPPolicy(KEY, width=386, layer_norm=True, in_dim=2), th, y0, KEY, ep, RewardConfig(), greedy=True)
    seen, dose = np.asarray(traj["seen"]), np.asarray(traj["dose"])
    for i in np.flatnonzero(~seen):
        assert dose[i] == dose[i - 1]


def test_lookahead_arms_are_parameter_matched_within_two_point_one_percent():
    def q(p, parts):
        return sum(x.size for x in jax.tree_util.tree_leaves(parts(p)) if hasattr(x, "size"))

    mlp = MLPPolicy(KEY, width=386, layer_norm=True, in_dim=2)
    lstm = LSTMPolicy(KEY, hidden=25, layer_norm=True, dt_input=True, in_dim=2)
    ode = ODERNNPolicy(KEY, width=62, layer_norm=True, in_dim=2)
    n_mlp = q(mlp, lambda p: (p.layers, p.norm))
    n_lstm = q(lstm, lambda p: (p.cell, p.head, p.norm))
    n_ode = q(ode, lambda p: (p.vf, p.cell, p.head, p.norm))
    assert n_mlp == 3091
    assert all(abs(n / n_mlp - 1) < 0.021 for n in (n_lstm, n_ode))
