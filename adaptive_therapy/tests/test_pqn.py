"""PQN invariants: targets, episode-level minibatching, exploration, matching."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy
from adaptive_therapy.rl.pqn import PQNConfig, episode_minibatches, q_lambda_targets, train
from adaptive_therapy.rl.rollout import EpisodeConfig, rollout

R = jnp.array([1.0, 2.0, 3.0, 4.0])
Q = jnp.array([[0.0, 5.0], [7.0, 1.0], [2.0, 3.0], [9.0, 9.0]])


def test_q_lambda_with_lambda_zero_is_one_step_q_learning():
    cfg = PQNConfig(gamma=0.9, lam=0.0)
    expected = jnp.array([1.0 + 0.9 * 7.0, 2.0 + 0.9 * 3.0, 3.0 + 0.9 * 9.0, 4.0])
    assert jnp.allclose(q_lambda_targets(R, Q, cfg), expected)


def test_q_lambda_with_lambda_one_is_the_discounted_monte_carlo_return():
    cfg = PQNConfig(gamma=0.9, lam=1.0)
    g3 = 4.0
    g2 = 3.0 + 0.9 * g3
    g1 = 2.0 + 0.9 * g2
    g0 = 1.0 + 0.9 * g1
    assert jnp.allclose(q_lambda_targets(R, Q, cfg), jnp.array([g0, g1, g2, g3]))


def test_minibatches_are_whole_episodes_never_shuffled_transitions():
    # A recurrent Q-value at visit t depends on visits 0..t, so a minibatch of
    # shuffled transitions (the paper's feedforward Alg. 2) would be wrong here.
    obs = jnp.arange(8 * 5, dtype=jnp.float32).reshape(8, 5)
    mbs = episode_minibatches({"obs": obs}, jax.random.PRNGKey(0), 4)
    assert mbs["obs"].shape == (4, 2, 5)
    rows = {tuple(r.tolist()) for r in mbs["obs"].reshape(8, 5)}
    assert rows == {tuple(r.tolist()) for r in obs}


def _actions(eps, greedy=False, n=24):
    c = sample_cohort("lotka_volterra", 1, seed=0)
    ep = EpisodeConfig(n_visits=n, model_id="lotka_volterra")
    pol = MLPPolicy(jax.random.PRNGKey(0), width=441, layer_norm=True)
    traj = rollout(
        pol,
        jnp.asarray(c.thetas[0]),
        jnp.asarray(c.y0s[0]),
        jax.random.PRNGKey(3),
        ep,
        RewardConfig(),
        greedy=greedy,
        eps=eps,
    )
    return traj["action"]


def test_epsilon_one_explores_every_dose_and_greedy_ignores_epsilon():
    assert set(_actions(1.0, n=40).tolist()) == {0, 1, 2}
    assert jnp.array_equal(_actions(1.0, greedy=True), _actions(0.0))


def test_the_two_q_networks_are_parameter_matched_excluding_the_unused_critic():
    def q_params(p):
        p = eqx.tree_at(lambda m: m.critic, p, None)
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(p, eqx.is_array)))

    k = jax.random.PRNGKey(0)
    mlp = q_params(MLPPolicy(k, width=441, layer_norm=True))
    ode = q_params(ODERNNPolicy(k, layer_norm=True))
    assert abs(mlp - ode) / ode < 0.01


@pytest.mark.parametrize("arch", ["mlp", "ode_rnn"])
def test_pqn_compiles_a_fixed_number_of_times_regardless_of_update_count(arch):
    import jax.monitoring as mon

    n = [0]

    def count(event, *_a, **_k):
        n[0] += event == "/jax/core/compile/backend_compile_duration"

    make = {
        "mlp": lambda: MLPPolicy(jax.random.PRNGKey(0), width=441, layer_norm=True),
        "ode_rnn": lambda: ODERNNPolicy(jax.random.PRNGKey(0), layer_norm=True),
    }[arch]
    c = sample_cohort("lotka_volterra", 8, seed=0)
    ep = EpisodeConfig(n_visits=4, model_id="lotka_volterra")
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    compiles = {}
    mon.register_event_duration_secs_listener(count)
    try:
        for u in (2, 2, 6):  # first call warms eager-op caches
            n[0] = 0
            cfg = PQNConfig(n_updates=u, batch_patients=4, n_epochs=1, n_minibatches=2)
            train(make(), th, y0, jax.random.PRNGKey(1), cfg, ep)
            compiles[u] = n[0]
    finally:
        mon.unregister_event_duration_listener(count)
    assert compiles[6] == compiles[2]
