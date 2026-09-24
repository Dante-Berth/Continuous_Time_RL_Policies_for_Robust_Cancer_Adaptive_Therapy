"""Tier-4 invariants: what a missed visit does, to the rollout, the arms and PQN.

PREREG W4: a missed visit removes the observation and the decision, never the
drug -- the last chosen dose is given again, the policy's state is carried
unchanged, and the next visit's dt is the true gap. If any of these silently
broke, the W4 comparison would measure the bug rather than the architecture.
"""

from __future__ import annotations

import equinox as eqx
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
from adaptive_therapy.rl import ppo
from adaptive_therapy.rl.pqn import PQNConfig, q_lambda_targets, train
from adaptive_therapy.rl.rollout import EpisodeConfig, replay, rollout, visit_mask

KEY = jax.random.PRNGKey(0)
ARCHS = {
    "mlp": lambda: MLPPolicy(KEY, width=441, layer_norm=True),
    "lstm": lambda: LSTMPolicy(KEY, layer_norm=True),
    "lstm_dt": lambda: LSTMPolicy(KEY, layer_norm=True, dt_input=True),
    "ode_rnn": lambda: ODERNNPolicy(KEY, layer_norm=True),
}


class _Probe(eqx.Module):
    """Records what it absorbs: state = (visits absorbed, last dt, total dt)."""

    field: int = eqx.field(static=True)

    def initial_state(self):
        return jnp.zeros((3,))

    def __call__(self, obs, state, dt, key=None):
        return jnp.zeros(3) + 0.0 * obs, jnp.stack([state[0] + 1.0, dt, state[2] + dt])

    def value(self, obs, state):
        return state[self.field]


def _one_patient():
    c = sample_cohort("lotka_volterra", 1, seed=0)
    return jnp.asarray(c.thetas[0]), jnp.asarray(c.y0s[0])


def test_the_first_visit_is_always_observed_and_the_drop_rate_is_honoured():
    masks = jax.vmap(lambda k: visit_mask(k, 52, 0.35))(jax.random.split(KEY, 2000))
    assert bool(masks[:, 0].all())
    assert abs(float(1.0 - masks[:, 1:].mean()) - 0.35) < 0.01
    assert bool(visit_mask(KEY, 52, 0.0).all())


def test_visits_missed_at_a_lower_rate_are_a_subset_of_those_missed_at_a_higher_one():
    for k in jax.random.split(KEY, 50):
        missed_20 = ~visit_mask(k, 52, 0.2)
        missed_50 = ~visit_mask(k, 52, 0.5)
        assert not bool((missed_20 & ~missed_50).any())


def test_a_missed_visit_leaves_the_policy_state_untouched():
    seen = jnp.array([True, False, False, True, True, False, True])
    ep = EpisodeConfig(n_visits=7)
    _, absorbed = replay(_Probe(0), jnp.zeros(7), ep, seen)
    assert jnp.array_equal(absorbed, jnp.cumsum(seen).astype(absorbed.dtype))


def test_the_policy_sees_the_true_gap_since_its_last_observation():
    seen = jnp.array([True, False, False, True, True, False, True])
    ep = EpisodeConfig(n_visits=7, visit_interval=7.0)
    _, last_dt = replay(_Probe(1), jnp.zeros(7), ep, seen)
    # visit 0 keeps the regular convention dt = one interval; then 21, 7, 14.
    assert jnp.array_equal(last_dt[seen], jnp.array([7.0, 21.0, 7.0, 14.0]))
    _, total = replay(_Probe(2), jnp.zeros(7), ep, seen)
    assert float(total[-1]) == 7.0 * 7  # the gaps tile the episode exactly


def test_a_missed_visit_gives_the_last_chosen_dose_again():
    th, y0 = _one_patient()
    ep = EpisodeConfig(n_visits=40, drop_rate=0.5)
    pol = MLPPolicy(KEY, width=441, layer_norm=True)
    traj = rollout(pol, th, y0, jax.random.PRNGKey(3), ep, RewardConfig(), eps=1.0)
    seen, dose = np.asarray(traj["seen"]), np.asarray(traj["dose"])
    assert 0 < (~seen).sum() < seen.size
    missed = np.flatnonzero(~seen)
    assert np.array_equal(dose[missed], dose[missed - 1])
    assert len(set(dose[seen].tolist())) == 3  # random actions still vary where chosen


def test_observation_noise_does_not_depend_on_which_visits_are_missed():
    """The mask has its own key stream: at a fixed schedule, drop rate changes nothing.

    This is what keeps drop rate 0 bit-identical to every pre-W4 rollout.
    """
    th, y0 = _one_patient()
    pol = RulePolicy("MTD")  # same doses whatever is missed, so same tumour
    t0 = rollout(pol, th, y0, KEY, EpisodeConfig(n_visits=20), RewardConfig(), greedy=True)
    ep5 = EpisodeConfig(n_visits=20, drop_rate=0.5)
    t5 = rollout(pol, th, y0, KEY, ep5, RewardConfig(), greedy=True)
    assert not bool(t5["seen"].all())
    assert jnp.array_equal(t0["obs"], t5["obs"])


@pytest.mark.parametrize("arch", ["lstm_dt", "ode_rnn"])
def test_replay_reproduces_the_rollout_decisions_under_missed_visits(arch):
    """PQN's targets come from replay; they must be the acting network's values."""
    th, y0 = _one_patient()
    ep = EpisodeConfig(n_visits=20, drop_rate=0.5)
    pol = ARCHS[arch]()
    traj = rollout(pol, th, y0, jax.random.PRNGKey(5), ep, RewardConfig(), greedy=True)
    logits, values = replay(pol, traj["obs"], ep, traj["seen"])
    seen = traj["seen"]
    assert not bool(seen.all())
    assert jnp.array_equal(jnp.argmax(logits, -1)[seen], traj["action"][seen])
    assert jnp.allclose(values, traj["value"], atol=1e-5)


def test_the_ode_policy_flow_composes_across_a_missed_visit():
    """Continuous time: a 14-day gap is two 7-day flows, not a different map."""
    p = ODERNNPolicy(KEY)
    h = jax.random.normal(jax.random.PRNGKey(2), (p.hidden,))
    assert jnp.allclose(p.flow(p.flow(h, 7.0), 7.0), p.flow(h, 14.0), atol=1e-6)
    assert not jnp.allclose(p.flow(h, 7.0), p.flow(h, 14.0))


def test_the_lstm_with_dt_input_reads_the_gap_and_the_plain_lstm_does_not():
    h0 = LSTMPolicy(KEY).initial_state()
    for dt_input, depends in ((True, True), (False, False)):
        p = LSTMPolicy(KEY, layer_norm=True, dt_input=dt_input)
        q7, _ = p(jnp.array(0.1), h0, 7.0)
        q21, _ = p(jnp.array(0.1), h0, 21.0)
        assert (not jnp.allclose(q7, q21)) == depends


def test_the_lstm_with_dt_input_is_parameter_matched_to_the_other_arms():
    def q_params(p):
        p = eqx.tree_at(lambda m: m.critic, p, None)
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(p, eqx.is_array)))

    mlp = q_params(ARCHS["mlp"]())
    assert abs(q_params(ARCHS["lstm_dt"]()) - mlp) / mlp < 0.05  # PREREG W4


R = jnp.array([1.0, 2.0, 3.0, 4.0])
Q = jnp.array([[0.0, 5.0], [7.0, 1.0], [2.0, 3.0], [9.0, 9.0]])


def test_q_lambda_with_every_visit_observed_is_the_regular_target():
    cfg = PQNConfig(gamma=0.9, lam=0.5)
    all_seen = jnp.ones(4, dtype=bool)
    assert jnp.array_equal(q_lambda_targets(R, Q, cfg, all_seen), q_lambda_targets(R, Q, cfg))


def test_q_lambda_bootstraps_at_the_next_decision_never_at_a_missed_visit():
    cfg = PQNConfig(gamma=0.9, lam=0.0)
    seen = jnp.array([True, False, True, True])
    t = q_lambda_targets(R, Q, cfg, seen)
    # the semi-MDP one-step target from visit 0: two weekly rewards, then max Q(s_2)
    assert jnp.isclose(t[0], 1.0 + 0.9 * 2.0 + 0.81 * 3.0)
    q_other = Q.at[1].set(jnp.array([100.0, -100.0]))
    assert jnp.array_equal(q_lambda_targets(R, q_other, cfg, seen), t)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_pqn_trains_every_arm_finitely_with_missed_visits(arch):
    c = sample_cohort("lotka_volterra", 8, seed=0)
    ep = EpisodeConfig(n_visits=8, drop_rate=0.5)
    cfg = PQNConfig(n_updates=3, batch_patients=4, n_epochs=1, n_minibatches=2)
    pol, hist = train(
        ARCHS[arch](), jnp.asarray(c.thetas), jnp.asarray(c.y0s), jax.random.PRNGKey(1), cfg, ep
    )
    assert all(np.isfinite(h["loss"]) for h in hist)
    leaves = jax.tree_util.tree_leaves(eqx.filter(pol, eqx.is_array))
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)


def test_ppo_refuses_missed_visits_rather_than_training_on_them_wrongly():
    c = sample_cohort("lotka_volterra", 4, seed=0)
    with pytest.raises(NotImplementedError):
        ppo.train(
            MLPPolicy(KEY),
            jnp.asarray(c.thetas),
            jnp.asarray(c.y0s),
            KEY,
            ep=EpisodeConfig(n_visits=4, drop_rate=0.2),
        )
