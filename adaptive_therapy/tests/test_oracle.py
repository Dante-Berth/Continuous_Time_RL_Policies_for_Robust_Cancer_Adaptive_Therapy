"""The full-information reference (PREREG S2) never changes what the arms see."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.policies.rules import RulePolicy
from adaptive_therapy.rl.pqn import PQNConfig, train
from adaptive_therapy.rl.rollout import ORACLE_DIMS, EpisodeConfig, oracle_features, rollout


def _patient():
    c = sample_cohort("lotka_volterra", 1, seed=0, ranges="strobl2021")
    return jnp.asarray(c.thetas[0]), jnp.asarray(c.y0s[0])


def _traj(oracle: str):
    th, y0 = _patient()
    ep = EpisodeConfig(n_visits=12, oracle=oracle)
    # MTD ignores its input, so all three rollouts dose identically.
    return rollout(
        RulePolicy("MTD"), th, y0, jax.random.PRNGKey(5), ep, RewardConfig(), greedy=True
    )


def test_burden_only_is_the_default_observation():
    assert EpisodeConfig().oracle == "none"
    assert _traj("none")["obs"].shape == (12,)


@pytest.mark.parametrize("oracle", ["state", "full"])
def test_oracle_reference_sees_the_arms_noisy_burden_observation_first(oracle):
    plain, ref = _traj("none"), _traj(oracle)
    assert ref["obs"].shape == (12, 1 + ORACLE_DIMS[oracle])
    assert jnp.array_equal(ref["obs"][:, 0], plain["obs"])
    assert jnp.array_equal(ref["burden"], plain["burden"])


def test_oracle_state_features_recover_the_hidden_populations():
    th, y0 = _patient()
    b0 = y0.sum()
    s, r = np.exp(np.asarray(oracle_features(y0, th, b0, "state"))) * float(b0)
    assert s == pytest.approx(float(y0[0]), rel=1e-5)
    assert r == pytest.approx(float(y0[1]), rel=1e-5)


def test_mlp_default_input_is_one_dim_so_every_saved_arm_still_loads():
    pol = MLPPolicy(jax.random.PRNGKey(0), width=441, layer_norm=True)
    assert pol.layers[0].weight.shape == (441, 1)


def test_pqn_trains_the_full_information_reference_on_vector_observations():
    c = sample_cohort("lotka_volterra", 8, seed=1, ranges="strobl2021")
    ep = EpisodeConfig(n_visits=8, oracle="full")
    pol = MLPPolicy(jax.random.PRNGKey(0), width=32, layer_norm=True, in_dim=9)
    cfg = PQNConfig(n_updates=2, batch_patients=8)
    trained, hist = train(
        pol, jnp.asarray(c.thetas), jnp.asarray(c.y0s), jax.random.PRNGKey(1), cfg, ep
    )
    assert np.isfinite(hist[-1]["loss"])
    assert trained.layers[0].weight.shape == (32, 9)
