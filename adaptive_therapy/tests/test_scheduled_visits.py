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
    traj = rollout(
        MLPPolicy(KEY, width=386, layer_norm=True, in_dim=2), th, y0, KEY, ep, RewardConfig()
    )
    seen = np.asarray(traj["seen"])
    got = np.asarray(traj["obs"])[:, 1] * LOOKAHEAD_SCALE
    assert np.allclose(got, slots_to_next_visit(seen))
    assert traj["obs"].shape == (52, 2)


def test_a_missed_slot_repeats_the_last_dose_under_a_schedule():
    th, y0 = _patient()
    ep = EpisodeConfig(schedule=(1, 3), lookahead=True)
    traj = rollout(
        MLPPolicy(KEY, width=386, layer_norm=True, in_dim=2),
        th,
        y0,
        KEY,
        ep,
        RewardConfig(),
        greedy=True,
    )
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


def test_a_schedule_pool_gives_each_patient_one_of_its_schedules():
    """PREREG W9: the draw comes from the episode key, so patients differ, and the
    lookahead each one is shown matches the schedule it was given."""
    pool = ((1, 3), (2, 2), (1, 1, 2))
    th, y0 = _patient()
    ep = EpisodeConfig(schedule_pool=pool, lookahead=True)
    pol = MLPPolicy(KEY, width=386, layer_norm=True, in_dim=2)
    allowed = [schedule_mask(52, s) for s in pool]
    picked = set()
    for k in jax.random.split(KEY, 24):
        traj = rollout(pol, th, y0, k, ep, RewardConfig())
        seen = np.asarray(traj["seen"])
        which = [i for i, m in enumerate(allowed) if (m == seen).all()]
        assert len(which) == 1
        assert np.allclose(
            np.asarray(traj["obs"])[:, 1] * LOOKAHEAD_SCALE, slots_to_next_visit(seen)
        )
        picked.add(which[0])
    assert picked == {0, 1, 2}
    with pytest.raises(ValueError):
        rollout(pol, th, y0, KEY, EpisodeConfig(schedule=(1,), schedule_pool=pool), RewardConfig())


def test_clock_dropout_hides_the_clock_in_about_half_the_episodes_and_can_be_forced():
    """PREREG V9: the clock is hidden when the first observation is positive,
    carried in the state; `hide` and `show` pin it at test."""
    p = MLPPolicy(KEY, width=386, layer_norm=True, clock_dropout=True)
    assert p.initial_state().shape == (3,)
    s0 = p.initial_state()
    _, up = p(jnp.array(0.02), s0, 7.0)
    _, down = p(jnp.array(-0.02), s0, 7.0)
    assert float(up[1]) == 1.0 and float(down[1]) == 0.0 and float(up[2]) == 1.0
    _, later = p(jnp.array(-0.9), up, 7.0)  # the flag is fixed after visit 0
    assert float(later[1]) == 1.0 and float(later[0]) == 14.0
    hide, show = p.forced("hide"), p.forced("show")
    far = up.at[0].set(300.0)
    a, _ = hide(jnp.array(0.1), far, 7.0)
    b, _ = hide(jnp.array(0.1), up, 7.0)
    c, _ = show(jnp.array(0.1), far, 7.0)
    d, _ = show(jnp.array(0.1), up, 7.0)
    assert jnp.allclose(a, b) and not jnp.allclose(c, d)


def test_clock_dropout_arm_matches_the_positional_mlp_parameter_count():
    def q(p):
        return sum(
            x.size for x in jax.tree_util.tree_leaves((p.layers, p.norm)) if hasattr(x, "size")
        )

    pos = MLPPolicy(KEY, width=386, layer_norm=True, clock=True)
    cd = MLPPolicy(KEY, width=386, layer_norm=True, clock_dropout=True)
    assert q(pos) == q(cd) == 3091


def test_state_dropout_resets_memory_in_the_hidden_half_and_adds_no_parameters():
    """PREREG V10: hidden episodes see only the current observation; shown ones
    are the unchanged recurrent policy; the flag is set once, at visit 0."""
    from adaptive_therapy.policies.state_dropout import StateDropout

    for inner in (LSTMPolicy(KEY, layer_norm=True), ODERNNPolicy(KEY, layer_norm=True)):
        sd = StateDropout(inner)
        s0 = sd.initial_state()
        q0, up = sd(jnp.array(0.02), s0, 7.0)  # positive first obs -> hidden
        _, down = sd(jnp.array(-0.02), s0, 7.0)  # -> shown
        assert float(up[-2]) == 1.0 and float(down[-2]) == 0.0 and float(up[-1]) == 1.0
        # hidden: the past cannot matter; shown: it can
        a1, _ = sd(jnp.array(0.3), up, 7.0)
        a2, _ = sd(jnp.array(0.3), up.at[:-2].add(0.5), 7.0)
        assert jnp.allclose(a1, a2)
        b1, _ = sd(jnp.array(0.3), down, 7.0)
        b2, _ = sd(jnp.array(0.3), down.at[:-2].add(0.5), 7.0)
        assert not jnp.allclose(b1, b2)
        # shown reproduces the bare policy exactly
        bare, hn = inner(jnp.array(0.3), inner.initial_state(), 7.0)
        shown, sn = sd.forced("show")(jnp.array(0.3), s0, 7.0)
        assert jnp.allclose(bare, shown) and jnp.allclose(hn, sn[:-2])
        # forced overrides the flag in both directions
        hide, show = sd.forced("hide"), sd.forced("show")
        h1, _ = hide(jnp.array(0.3), down, 7.0)
        h2, _ = hide(jnp.array(0.3), down.at[:-2].add(0.5), 7.0)
        w1, _ = show(jnp.array(0.3), up, 7.0)
        w2, _ = show(jnp.array(0.3), up.at[:-2].add(0.5), 7.0)
        assert jnp.allclose(h1, h2) and not jnp.allclose(w1, w2)
        n = lambda t: sum(x.size for x in jax.tree_util.tree_leaves(t) if hasattr(x, "size"))  # noqa: E731
        assert n(sd.inner) == n(inner)
