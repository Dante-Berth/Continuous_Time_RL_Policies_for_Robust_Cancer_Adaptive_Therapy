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
