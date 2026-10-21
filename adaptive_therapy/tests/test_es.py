"""Evolution-strategies trainer (PREREG V14)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.policies.lstm import LSTMPolicy
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.rl.es import ESConfig, centred_ranks, es_gradient, train
from adaptive_therapy.rl.rollout import EpisodeConfig, batch_rollout


def test_centred_ranks_ignore_the_scale_and_outliers_of_the_fitness():
    f = jnp.array([3.0, -1.0, 1e9, 0.5])
    assert np.asarray(centred_ranks(f)).tolist() == pytest.approx([1 / 6, -0.5, 0.5, -1 / 6])
    assert np.allclose(centred_ranks(f), centred_ranks(f * 7 - 2))


def test_es_gradient_points_toward_the_noise_that_scored_higher():
    eps = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    # member 0 (+eps0) is best, its mirror (-eps0) worst; the pair 1 is in between
    fitness = jnp.array([3.0, 2.0, 0.0, 1.0])
    g = np.asarray(es_gradient(eps, fitness, sigma=0.5))
    assert g[0] > 0 and g[0] > abs(g[1])


def test_training_rejects_missed_visits_because_a_missed_visit_is_not_a_decision():
    c = sample_cohort("lotka_volterra", 4, seed=0)
    with pytest.raises(ValueError, match="regular visits"):
        train(
            MLPPolicy(jax.random.PRNGKey(0), width=8),
            jnp.asarray(c.thetas),
            jnp.asarray(c.y0s),
            jax.random.PRNGKey(1),
            ESConfig(n_gens=1),
            EpisodeConfig(n_visits=4, drop_rate=0.2),
        )


@pytest.mark.parametrize("arch", ["mlp", "lstm"])
def test_es_compiles_a_fixed_number_of_times_regardless_of_generation_count(arch):
    import jax.monitoring as mon

    n = [0]

    def count(event, *_a, **_k):
        n[0] += event == "/jax/core/compile/backend_compile_duration"

    make = {
        "mlp": lambda: MLPPolicy(jax.random.PRNGKey(0), width=16),
        "lstm": lambda: LSTMPolicy(jax.random.PRNGKey(0), layer_norm=True),
    }[arch]
    c = sample_cohort("lotka_volterra", 8, seed=0)
    ep = EpisodeConfig(n_visits=4, model_id="lotka_volterra")
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    compiles = {}
    mon.register_event_duration_secs_listener(count)
    try:
        for g in (2, 2, 6):  # first call warms eager-op caches
            n[0] = 0
            train(
                make(),
                th,
                y0,
                jax.random.PRNGKey(1),
                ESConfig(n_gens=g, pop=4, batch_patients=4),
                ep,
            )
            compiles[g] = n[0]
    finally:
        mon.unregister_event_duration_listener(count)
    assert compiles[6] == compiles[2]


def test_es_learns_to_prefer_dosing_when_dosing_is_all_the_reward():
    """A reward of dose-only (survival and penalty off) must drive the greedy dose up."""
    c = sample_cohort("lotka_volterra", 16, seed=0)
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    ep = EpisodeConfig(n_visits=6, model_id="lotka_volterra")
    rw = RewardConfig(exposure_penalty=-1.0)  # negative penalty = pays for drug
    pol, _ = train(
        MLPPolicy(jax.random.PRNGKey(0), width=16),
        th,
        y0,
        jax.random.PRNGKey(2),
        ESConfig(n_gens=60, pop=32, batch_patients=16),
        ep,
        rw,
    )
    traj = batch_rollout(pol, th, y0, jax.random.PRNGKey(3), ep, rw, greedy=True)
    assert float(traj["dose"].mean()) > 0.8
