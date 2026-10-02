"""Rule baselines: each dosing rule does what its name says."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.policies.rules import RulePolicy


def _doses(kind, burdens):
    """Greedy action index at each observed burden/baseline, carrying state."""
    p = RulePolicy(kind)
    h, out = p.initial_state(), []
    for b in burdens:
        q, h = p(jnp.log(jnp.asarray(b)), h, 7.0)
        out.append(int(jnp.argmax(q)))
    return out


def test_none_never_treats_and_mtd_always_does():
    b = [1.0, 0.3, 2.0, 0.9]
    assert _doses("none", b) == [0, 0, 0, 0]
    assert _doses("MTD", b) == [2, 2, 2, 2]


def test_threshold_rule_treats_only_above_eighty_percent_of_baseline():
    assert _doses("thr0.8", [0.5, 0.79, 0.81, 1.5]) == [0, 0, 2, 2]


def test_at50_holds_below_half_baseline_until_burden_returns_to_baseline():
    # on -> falls below 0.5 -> off; stays off while rising through 0.8; on again at 1.0
    assert _doses("AT50", [1.0, 0.7, 0.45, 0.6, 0.8, 1.0, 0.9]) == [2, 2, 0, 0, 0, 2, 2]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        RulePolicy("MTD50")


def test_rules_are_finite_in_a_real_rollout():
    import jax

    from adaptive_therapy.envs.reward import RewardConfig
    from adaptive_therapy.envs.sampling import sample_cohort
    from adaptive_therapy.rl.rollout import EpisodeConfig, batch_rollout

    c = sample_cohort("lotka_volterra", 4, seed=0)
    for kind in ("none", "MTD", "AT50", "thr0.8"):
        tr = batch_rollout(
            RulePolicy(kind),
            jnp.asarray(c.thetas),
            jnp.asarray(c.y0s),
            jax.random.PRNGKey(0),
            EpisodeConfig(),
            RewardConfig(),
            greedy=True,
        )
        assert np.isfinite(np.asarray(tr["burden"])).all()


def test_sequence_policy_plays_its_sequence_whatever_it_observes():
    from adaptive_therapy.policies.rules import SequencePolicy

    seq = [2, 0, 1, 1, 0]
    p, h, out = SequencePolicy(seq), None, []
    h = p.initial_state()
    for b in [0.1, 5.0, 1.0, 0.01, 2.0]:
        q, h = p(jnp.log(jnp.asarray(b)), h, 7.0)
        out.append(int(jnp.argmax(q)))
    assert out == seq
