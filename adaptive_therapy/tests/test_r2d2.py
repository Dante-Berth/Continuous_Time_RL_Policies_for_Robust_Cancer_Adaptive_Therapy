"""Replay-trainer invariants: n-step targets, ring buffer, compile count, learning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.policies.lstm import LSTMPolicy
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.rl.r2d2 import R2D2Config, buffer_insert, n_step_double_q_targets, train
from adaptive_therapy.rl.rollout import EpisodeConfig

R = jnp.array([1.0, 2.0, 3.0, 4.0])
Q_ON = jnp.array([[0.0, 5.0], [7.0, 1.0], [2.0, 3.0], [9.0, 8.0]])
Q_TG = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_one_step_target_bootstraps_the_target_net_at_the_online_nets_argmax():
    cfg = R2D2Config(gamma=0.9, n_step=1)
    # t=0: online argmax at s_1 is action 0 -> target value 1.0; t=2: argmax at s_3 is 0 -> 5.0
    expected = jnp.array([1.0 + 0.9 * 1.0, 2.0 + 0.9 * 4.0, 3.0 + 0.9 * 5.0, 4.0])
    assert jnp.allclose(n_step_double_q_targets(R, Q_ON, Q_TG, cfg), expected)


def test_n_step_target_with_n_beyond_the_horizon_is_the_discounted_return_with_no_bootstrap():
    cfg = R2D2Config(gamma=0.9, n_step=4)
    g3 = 4.0
    g2 = 3.0 + 0.9 * g3
    g1 = 2.0 + 0.9 * g2
    expected = jnp.array([1.0 + 0.9 * g1, g1, g2, g3])
    # t=0 reaches s_4, which is past the end: no bootstrap anywhere.
    assert jnp.allclose(n_step_double_q_targets(R, Q_ON, Q_TG, cfg), expected)


def test_ring_buffer_overwrites_the_oldest_episodes_first():
    buf = {"x": jnp.zeros((4, 2))}
    ptr = 0
    for i in (1.0, 2.0, 3.0):
        buf, ptr = buffer_insert(buf, {"x": jnp.full((2, 2), i)}, ptr, 4)
    assert np.asarray(buf["x"][:, 0]).tolist() == [3.0, 3.0, 2.0, 2.0]
    assert int(ptr) == 2


def test_training_rejects_missed_visits_because_a_missed_visit_is_not_a_decision():
    c = sample_cohort("lotka_volterra", 4, seed=0)
    with pytest.raises(ValueError, match="regular visits"):
        train(
            MLPPolicy(jax.random.PRNGKey(0), width=441, layer_norm=True),
            jnp.asarray(c.thetas),
            jnp.asarray(c.y0s),
            jax.random.PRNGKey(1),
            R2D2Config(n_updates=1),
            EpisodeConfig(n_visits=4, drop_rate=0.2),
        )


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_r2d2_compiles_a_fixed_number_of_times_regardless_of_update_count(arch):
    import jax.monitoring as mon

    n = [0]

    def count(event, *_a, **_k):
        n[0] += event == "/jax/core/compile/backend_compile_duration"

    make = {
        "mlp": lambda: MLPPolicy(jax.random.PRNGKey(0), width=441, layer_norm=True),
        "lstm": lambda: LSTMPolicy(jax.random.PRNGKey(0), layer_norm=True),
    }[arch]
    c = sample_cohort("lotka_volterra", 8, seed=0)
    ep = EpisodeConfig(n_visits=4, model_id="lotka_volterra")
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    compiles = {}
    mon.register_event_duration_secs_listener(count)
    try:
        for u in (2, 2, 6):  # first call warms eager-op caches
            n[0] = 0
            cfg = R2D2Config(
                n_updates=u, batch_patients=4, grads_per_update=2, batch_episodes=4, capacity=8
            )
            train(make(), th, y0, jax.random.PRNGKey(1), cfg, ep)
            compiles[u] = n[0]
    finally:
        mon.unregister_event_duration_listener(count)
    assert compiles[6] == compiles[2]


def test_r2d2_learns_to_prefer_dosing_when_dosing_is_all_the_reward():
    """A reward of dose-only (survival and penalty off) must drive the greedy dose up."""
    from adaptive_therapy.envs.reward import RewardConfig
    from adaptive_therapy.rl.rollout import batch_rollout

    c = sample_cohort("lotka_volterra", 16, seed=0)
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    ep = EpisodeConfig(n_visits=6, model_id="lotka_volterra")
    rw = RewardConfig(exposure_penalty=-1.0)  # negative penalty = pays for drug
    pol, _ = train(
        MLPPolicy(jax.random.PRNGKey(0), width=64, layer_norm=True),
        th,
        y0,
        jax.random.PRNGKey(2),
        R2D2Config(n_updates=60, batch_patients=16, grads_per_update=4, batch_episodes=16),
        ep,
        rw,
    )
    traj = batch_rollout(pol, th, y0, jax.random.PRNGKey(3), ep, rw, greedy=True)
    assert float(traj["dose"].mean()) > 0.8
