"""Invariants of the policy arms and the rollout loop.

The parameter-matching test is the one that protects the Phase 2 conclusion:
if the arms drift apart in capacity, an ODE win becomes uninterpretable.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.policies.base import DOSE_LEVELS, N_ACTIONS, normalise_obs
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy
from adaptive_therapy.rl.rollout import EpisodeConfig, rollout

KEY = jax.random.PRNGKey(0)


def n_params(p) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(p, eqx.is_array)))


def test_the_two_arms_are_parameter_matched():
    """PREREG [2026-09-22] + AMENDMENT: counts must be within 2x.

    They are exactly equal at the declared widths. If a future change breaks
    that, the architecture comparison is confounded by capacity and this must
    fail loudly rather than be discovered at analysis time.
    """
    mlp, ode = n_params(MLPPolicy(KEY)), n_params(ODERNNPolicy(KEY))
    assert mlp == ode == 3076
    assert 0.5 <= ode / mlp <= 2.0


def test_both_arms_expose_the_same_interface():
    """The RL loop must not special-case an architecture."""
    for p in (MLPPolicy(KEY), ODERNNPolicy(KEY)):
        s = p.initial_state()
        logits, s2 = p(jnp.array(0.1), s, 7.0)
        assert logits.shape == (N_ACTIONS,)
        assert s2.shape == s.shape
        assert jnp.isfinite(p.value(jnp.array(0.1), s))


def test_ode_policy_state_depends_on_the_gap_length():
    """The architectural claim under test: dt must actually matter.

    If the ODE arm ignored dt it would be an expensive GRU, and the Tier-4
    experiment would be measuring nothing.
    """
    p = ODERNNPolicy(KEY)
    s = p.initial_state() + 0.3
    _, short = p(jnp.array(0.5), s, 1.0)
    _, long = p(jnp.array(0.5), s, 30.0)
    assert not jnp.allclose(short, long)


def test_mlp_is_memoryless_by_construction():
    """The control: the MLP must NOT carry state, or the arms differ in two
    ways at once and the ablation is not isolating."""
    p = MLPPolicy(KEY)
    assert p.initial_state().shape == (0,)
    a, _ = p(jnp.array(0.5), p.initial_state(), 1.0)
    b, _ = p(jnp.array(0.5), p.initial_state(), 99.0)
    assert jnp.allclose(a, b)  # dt cannot influence a memoryless policy


def test_positional_mlp_sees_elapsed_time_but_not_past_observations():
    """PREREG V7: the clock is its only state. Same obs at a later time may
    differ; a different past observation at the same time may not."""
    p = MLPPolicy(KEY, width=386, layer_norm=True, clock=True)
    s0 = p.initial_state()
    _, s1 = p(jnp.array(0.9), s0, 7.0)
    _, s1b = p(jnp.array(-0.4), s0, 7.0)
    assert jnp.allclose(s1, s1b) and float(s1[0]) == 7.0
    a, _ = p(jnp.array(0.2), s1, 7.0)
    b, _ = p(jnp.array(0.2), s1 + 200.0, 7.0)
    assert not jnp.allclose(a, b)


def test_positional_mlp_q_path_matches_the_other_arms():
    """V7 width 386 keeps the Q-path within 1% of the MLP's 3090."""

    def q_path(p):
        return n_params((p.layers, p.norm))

    mlp = q_path(MLPPolicy(KEY, width=441, layer_norm=True))
    pos = q_path(MLPPolicy(KEY, width=386, layer_norm=True, clock=True))
    assert mlp == 3090 and abs(pos / mlp - 1) < 0.01


def test_normalised_observation_is_scale_free():
    """A Tier-1 shift moves K; the policy must not break on scale alone."""
    assert float(normalise_obs(jnp.array(200.0), jnp.array(100.0))) == pytest.approx(
        float(normalise_obs(jnp.array(2e6), jnp.array(1e6))), rel=1e-6
    )


def test_doses_are_discrete_boluses():
    """PROJECT.md 7: not continuous infusion."""
    assert DOSE_LEVELS.shape == (N_ACTIONS,)
    assert float(DOSE_LEVELS[0]) == 0.0
    assert float(DOSE_LEVELS[-1]) == 1.0


@pytest.mark.parametrize("arch", [MLPPolicy, ODERNNPolicy])
def test_rollout_produces_finite_trajectories(arch):
    c = sample_cohort("lotka_volterra", 1, seed=0)
    t = rollout(
        arch(KEY),
        jnp.asarray(c.thetas[0]),
        jnp.asarray(c.y0s[0]),
        KEY,
        EpisodeConfig(),
        RewardConfig(),
    )
    for k in ("burden", "dose", "reward", "logp"):
        assert jnp.isfinite(t[k]).all(), f"{k} not finite"
    assert (t["burden"] > 0).all()
    assert jnp.isin(t["dose"], DOSE_LEVELS).all()


def test_treatment_reduces_burden_relative_to_no_treatment():
    """Sanity on the closed loop: dosing must actually do something.

    A loop where the action does not reach the simulator would train happily
    and produce a null result for entirely the wrong reason.
    """
    from adaptive_therapy.envs.base import PKConfig

    c = sample_cohort("lotka_volterra", 1, seed=0)
    th, y0 = jnp.asarray(c.thetas[0]), jnp.asarray(c.y0s[0])

    class Always(eqx.Module):
        level: int

        def initial_state(self):
            return jnp.zeros((0,))

        def __call__(self, obs, state, dt, key=None):
            return jnp.array([-1e6, -1e6, -1e6]).at[self.level].set(1e6), state

        def value(self, obs, state):
            return jnp.array(0.0)

    ep, rw, pk = EpisodeConfig(), RewardConfig(), PKConfig()
    none = rollout(Always(0), th, y0, KEY, ep, rw, pk, greedy=True)
    full = rollout(Always(2), th, y0, KEY, ep, rw, pk, greedy=True)
    assert float(full["burden"][-1]) < float(none["burden"][-1])
    assert float(full["dose"].mean()) > float(none["dose"].mean())


def test_ppo_compiles_a_fixed_number_of_times_regardless_of_update_count():
    # 2026-09-23: the old-logp rollout ran eagerly and recompiled every update;
    # at 300 updates the JIT mappings crossed vm.max_map_count and crashed every
    # Phase 2 cell with "LLVM ERROR: Unable to allocate section memory".
    import jax.monitoring as mon

    from adaptive_therapy.rl.ppo import PPOConfig, train

    n = [0]

    def count(event, *_a, **_k):
        n[0] += event == "/jax/core/compile/backend_compile_duration"

    c = sample_cohort("lotka_volterra", 8, seed=0)
    ep = EpisodeConfig(n_visits=4, model_id="lotka_volterra")
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    compiles = {}
    mon.register_event_duration_secs_listener(count)
    try:
        for u in (2, 2, 6):  # first call warms eager-op caches
            n[0] = 0
            cfg = PPOConfig(n_updates=u, batch_patients=4, n_epochs=1)
            train(MLPPolicy(jax.random.PRNGKey(0)), th, y0, jax.random.PRNGKey(1), cfg, ep)
            compiles[u] = n[0]
    finally:
        mon.unregister_event_duration_listener(count)
    assert compiles[6] == compiles[2]


@pytest.mark.parametrize("arch", [MLPPolicy, ODERNNPolicy])
def test_replaying_the_observed_sequence_reproduces_the_rollout_log_probs(arch):
    from adaptive_therapy.rl.rollout import replay

    c = sample_cohort("lotka_volterra", 1, seed=0)
    ep = EpisodeConfig(n_visits=6, model_id="lotka_volterra")
    pol = arch(jax.random.PRNGKey(0))
    traj = rollout(
        pol,
        jnp.asarray(c.thetas[0]),
        jnp.asarray(c.y0s[0]),
        jax.random.PRNGKey(1),
        ep,
        RewardConfig(),
    )
    logits, values = replay(pol, traj["obs"], ep)
    logp = jax.nn.log_softmax(logits)[jnp.arange(ep.n_visits), traj["action"]]
    assert jnp.allclose(logp, traj["logp"], atol=1e-5)
    assert jnp.allclose(values, traj["value"], atol=1e-5)


class _FixedSchedule(eqx.Module):
    """A policy that ignores its observation: an open-loop schedule."""

    def initial_state(self):
        return jnp.zeros((1,))

    def __call__(self, obs, state, dt, key=None):
        return jnp.array([0.0, 2.0, 0.0]) + 0.0 * obs, state + 1.0

    def value(self, obs, state):
        return jnp.asarray(0.0)


def test_an_open_loop_schedule_has_zero_observation_dependence_and_a_policy_does_not():
    from adaptive_therapy.rl.rollout import observation_dependence

    ep = EpisodeConfig(n_visits=5, model_id="lotka_volterra")
    obs = jax.random.normal(jax.random.PRNGKey(0), (16, 5))
    k = jax.random.PRNGKey(1)
    assert observation_dependence(_FixedSchedule(), obs, ep, k) == 0.0
    assert observation_dependence(MLPPolicy(jax.random.PRNGKey(2)), obs, ep, k) > 0.0


@pytest.mark.parametrize("arch", [MLPPolicy, ODERNNPolicy])
def test_every_arms_critic_sees_the_current_observation(arch):
    # 2026-09-23: the ODE-RNN critic read the pre-observation hidden state, so
    # its value at visit k ignored visit k's measurement; the MLP's did not.
    from adaptive_therapy.rl.rollout import replay

    ep = EpisodeConfig(n_visits=4, model_id="lotka_volterra")
    pol = arch(jax.random.PRNGKey(0))
    obs = jnp.array([0.0, 0.1, -0.2, 0.3])
    _, v_a = replay(pol, obs, ep)
    _, v_b = replay(pol, obs.at[-1].set(2.0), ep)
    assert not jnp.isclose(v_a[-1], v_b[-1])


def test_the_entropy_bonus_is_the_exact_policy_entropy_and_pushes_toward_uniform():
    from adaptive_therapy.rl.ppo import _entropy

    assert jnp.isclose(_entropy(jnp.zeros(3)), jnp.log(3.0))
    peaked = jnp.array([4.0, 0.0, 0.0])
    grad = jax.grad(_entropy)(peaked)
    assert grad[0] < 0 and grad[1] > 0  # ascent flattens the dominant action


@pytest.mark.parametrize("arch", [MLPPolicy, ODERNNPolicy])
def test_an_untrained_policy_is_still_exploring_at_the_end_of_an_episode(arch):
    # 2026-09-23: the default-init ODE-RNN's latent flow drove h to a saturated
    # fixed point, so it was deterministic (entropy 0.00) by visit 51 before
    # any training -- an open-loop schedule PPO never escaped.
    from adaptive_therapy.rl.rollout import replay

    c = sample_cohort("lotka_volterra", 8, seed=0)
    ep = EpisodeConfig(model_id="lotka_volterra")  # the full 52 visits
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    for s in range(3):
        pol = arch(jax.random.PRNGKey(s))
        traj = jax.vmap(lambda t, y, k: rollout(pol, t, y, k, ep, RewardConfig()))(
            jnp.asarray(c.thetas), jnp.asarray(c.y0s), keys
        )
        logits = jax.vmap(lambda o: replay(pol, o, ep)[0])(traj["obs"])
        p = jax.nn.softmax(logits[:, -1], axis=-1)
        entropy = -(p * jnp.log(p)).sum(-1).mean()
        assert entropy > 0.9, f"seed {s}: entropy {entropy:.3f} at the last visit"
