"""Closed-loop episode rollout: policy <-> simulator, burden-only.

One episode = a virtual patient observed at monitoring visits. At each visit the
policy sees ONLY the noisy scalar burden (PROJECT.md 7), picks a dose, and the
simulator integrates to the next visit under that dose held constant -- boluses
with exponential PK, not continuous control.

The loop is IDENTICAL for every architecture. Everything that differs between
arms lives inside the policy object, which is what makes the Phase 2/3 ablation
an architecture comparison rather than a comparison of two systems.

Missed visits (Tier 4, PREREG W4): the simulator still steps weekly, but a
missed visit gives the policy no observation and no decision. Its state is
carried unchanged, the gap it will see at the next visit grows by one interval,
and the last chosen dose is given again ("hold") -- a missed visit loses
information, not drug. TTP and exposure are still scored on the weekly
ground-truth grid, so every drop rate is measured on the same clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from adaptive_therapy.envs.base import ModelId, PKConfig, SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_one_patient
from adaptive_therapy.envs.pk import administer, decay, effect
from adaptive_therapy.envs.reward import RewardConfig, progression_update, step_reward
from adaptive_therapy.envs.sampling import STROBL_R_S
from adaptive_therapy.policies.base import DOSE_LEVELS, N_ACTIONS, normalise_obs


@dataclass(frozen=True)
class EpisodeConfig:
    n_visits: int = 52
    visit_interval: float = 7.0  # days
    obs_noise: float = 0.05
    model_id: ModelId = "lotka_volterra"
    drop_rate: float = 0.0  # P(a visit after the first is missed); Tier 4
    # "none" (every arm), or a REFERENCE that also sees the hidden state:
    # "state" adds S, R; "full" adds theta too. 2026-09-29, owner-approved
    # exception to burden-only for PREREG S2's headroom bound -- never an arm,
    # never reported as a policy. See oracle_features.
    oracle: str = "none"
    # What a missed visit gives: "hold" repeats the last chosen dose (every run
    # up to S3); "full" gives full dose. 2026-09-30, PREREG M1's protocol question.
    miss_dose: str = "hold"


ORACLE_DIMS = {"none": 0, "state": 2, "full": 8}


def oracle_features(y, theta, baseline, level: str) -> jnp.ndarray:
    """The hidden quantities a burden-only policy never sees, scaled for a network.

    THIS DELIBERATELY CROSSES THE POMDP BOUNDARY (envs/base.observe). It exists
    only for PREREG S2's full-information reference: "state" -> noise-free
    log(S/b0), log(R/b0); "full" adds theta as log(r_S/r0), log(r_R/r0),
    log(K/b0), d_S/r0, d_R/r0, d_D with r0 = Strobl's r_S. LV only.
    """
    b0 = jnp.maximum(baseline, 1e-10)
    feats = [jnp.log(jnp.maximum(y[0], 1e-10) / b0), jnp.log(jnp.maximum(y[1], 1e-10) / b0)]
    if level == "full":
        r_S, r_R, K, d_S, d_R, d_D = (theta[i] for i in range(6))
        feats += [
            jnp.log(jnp.maximum(r_S, 1e-10) / STROBL_R_S),
            jnp.log(jnp.maximum(r_R, 1e-10) / STROBL_R_S),
            jnp.log(jnp.maximum(K, 1e-10) / b0),
            d_S / STROBL_R_S,
            d_R / STROBL_R_S,
            d_D,
        ]
    return jnp.stack(feats)


# The visit mask is drawn from fold_in(key, _MASK_STREAM), never from the key
# chain that draws observation noise and exploration: at drop_rate 0 every
# earlier rollout (W3c included) is reproduced exactly, and at any rate the
# noise a patient sees is the same whichever visits are missed.
_MASK_STREAM = 4


def visit_mask(key, n_visits: int, drop_rate: float) -> jnp.ndarray:
    """(n_visits,) bool, True where the visit happens. Visit 0 always does.

    One uniform draw per visit, observed iff u >= drop_rate: for a fixed key the
    visits missed at 20% are a subset of those missed at 50%, so the drop-rate
    conditions are nested and paired, not independent resamples.
    """
    u = jax.random.uniform(key, (n_visits,))
    return (u >= drop_rate).at[0].set(True)


def _visit(policy, h, since, obs, seen, interval: float):
    """One monitoring slot -> (logits, h', since'). The ONE home of missed-visit logic.

    `since` is the time since the last observed visit, the `dt` the policy
    integrates (it starts at one interval, so a regular episode passes dt =
    interval at every visit, as before W4). The policy runs at every slot --
    under vmap both branches of a cond would run anyway -- and at a missed one
    its state update is discarded; its logits there must be ignored.
    """
    logits, h_obs = policy(obs, h, since)
    h2 = jnp.where(seen, h_obs, h)
    since2 = jnp.where(seen, interval, since + interval)
    return logits, h2, since2


def rollout(
    policy,
    theta: jnp.ndarray,
    y0: jnp.ndarray,
    key,
    ep: EpisodeConfig,
    rw: RewardConfig,
    pk: PKConfig | None = None,
    greedy: bool = False,
    eps=None,
):
    """Run one patient. Returns per-visit trajectories for training and eval.

    Actions: sampled from softmax(logits) (PPO), argmax if `greedy`, or
    epsilon-greedy over the outputs read as Q-values when `eps` is given (PQN).
    At a missed visit (`seen` False in the output) the action is the held one.
    """
    pk = pk or PKConfig()
    cfg = SolverConfig(t0=0.0, t1=ep.visit_interval, dt0=0.05, n_save=2)
    if ep.oracle not in ORACLE_DIMS:
        raise ValueError(f"unknown oracle {ep.oracle!r}; expected one of {tuple(ORACLE_DIMS)}")
    if ep.oracle != "none" and ep.model_id != "lotka_volterra":
        raise ValueError("oracle observations exist for lotka_volterra only")
    if ep.miss_dose not in ("hold", "full"):
        raise ValueError(f"unknown miss_dose {ep.miss_dose!r}; expected 'hold' or 'full'")

    baseline = observe(y0, ep.model_id)

    def step(carry, seen):
        state, y, conc, h, nadir, k, since, held, run, progressed = carry
        k, k_obs, k_act = jax.random.split(k, 3)

        true_b = observe(y, ep.model_id)
        noisy = true_b * jnp.exp(jax.random.normal(k_obs) * ep.obs_noise)
        obs = normalise_obs(noisy, baseline)
        if ep.oracle != "none":  # S2's reference only; the burden obs stays first
            obs = jnp.concatenate([obs[None], oracle_features(y, theta, baseline, ep.oracle)])

        logits, h2, since2 = _visit(policy, h, since, obs, seen, ep.visit_interval)
        if eps is None:
            chosen = jnp.where(
                greedy,
                jnp.argmax(logits),
                jax.random.categorical(k_act, logits),
            )
        else:
            k_coin, k_rand = jax.random.split(k_act)
            explore = jax.random.uniform(k_coin) < eps
            random_a = jax.random.randint(k_rand, (), 0, logits.shape[-1])
            chosen = jnp.where(explore & ~greedy, random_a, jnp.argmax(logits))
        missed = held if ep.miss_dose == "hold" else jnp.array(N_ACTIONS - 1)
        action = jnp.where(seen, chosen, missed)
        logp = jax.nn.log_softmax(logits)[action]
        dose = DOSE_LEVELS[action]

        conc2 = administer(conc, dose, pk)
        u = effect(conc2)
        y2 = solve_one_patient(ep.model_id, theta, y, u, cfg)[-1]
        conc3 = decay(conc2, ep.visit_interval, pk)

        nadir2 = jnp.minimum(nadir, true_b)
        run2, progressed2 = progression_update(
            run, progressed, true_b, nadir2, ep.visit_interval, rw
        )
        r = step_reward(true_b, nadir2, dose, ep.visit_interval, rw, progressed2)

        out = {
            "burden": true_b,
            "dose": dose,
            "reward": r,
            "logp": logp,
            "action": action,
            "obs": obs,
            "seen": seen,
            # AFTER this visit's observation is absorbed (h2), so every arm's
            # critic sees the current measurement. 2026-09-23: it read the
            # pre-observation state h, which left the ODE-RNN's critic one
            # visit stale while the MLP's read obs directly -- an arm
            # asymmetry the Phase 2 comparison did not declare (L312).
            "value": policy.value(obs, h2),
        }
        return (state, y2, conc3, h2, nadir2, k, since2, action, run2, progressed2), out

    seen = visit_mask(jax.random.fold_in(key, _MASK_STREAM), ep.n_visits, ep.drop_rate)
    init = (
        None,
        y0,
        jnp.array(0.0),
        policy.initial_state(),
        observe(y0, ep.model_id),
        key,
        jnp.array(ep.visit_interval),
        jnp.array(0),  # visit 0 is always observed, so this is never given
        jnp.array(0),  # visits in the current run above 1.2 x nadir
        jnp.array(False),  # progression confirmed online (RewardConfig.survival_until)
    )
    _, traj = jax.lax.scan(step, init, seen)
    return traj


def batch_rollout(policy, thetas, y0s, key, ep, rw, greedy=False, eps=None):
    """`rollout` vmapped over a cohort, one PRNG key per patient."""
    keys = jax.random.split(key, thetas.shape[0])
    return jax.vmap(lambda th, y, k: rollout(policy, th, y, k, ep, rw, greedy=greedy, eps=eps))(
        thetas, y0s, keys
    )


def replay(policy, obs: jnp.ndarray, ep: EpisodeConfig, seen: jnp.ndarray | None = None):
    """Re-run the policy over a STORED observation sequence, no simulator.

    `obs` (T,) is what one patient's rollout observed, `seen` (T,) which of its
    visits happened (default: all). Returns `(logits (T, A), values (T,))`, the
    value read AFTER visit k's observation is absorbed -- so every arm's critic
    sees the current measurement. This is how recurrent PPO evaluates stored
    actions under new parameters, and how a trained policy is probed for
    whether it listens to its input at all. Logits at missed visits are
    meaningless (no decision was made there).
    """
    seen = jnp.ones(obs.shape[0], dtype=bool) if seen is None else seen

    def step(carry, x):
        h, since = carry
        o, s = x
        logits, h2, since2 = _visit(policy, h, since, o, s, ep.visit_interval)
        return (h2, since2), (logits, policy.value(o, h2))

    init = (policy.initial_state(), jnp.array(ep.visit_interval))
    _, (logits, values) = jax.lax.scan(step, init, (obs, seen))
    return logits, values


def observation_dependence(
    policy, obs: jnp.ndarray, ep: EpisodeConfig, key, greedy: bool = False
) -> float:
    """How much the action distribution depends on WHICH patient is observed.

    Each patient's observation history is swapped for another patient's, and
    the total-variation distance between the two action distributions is
    averaged over patients and visits. 0 means an open-loop schedule -- the
    same dosing for every patient whatever their tumour does; that is not a
    feedback policy. Added 2026-09-23 after Phase 2's ODE-RNN arm went open-loop
    in 4/6 cells unnoticed (RESEARCH_LOG L312).

    `greedy=True` scores the argmax action instead (a Q-network acts greedily,
    and softmax of Q-values is not its policy): the result is then the fraction
    of patient-visits whose greedy dose changes when the history is swapped.
    """
    perm = jax.random.permutation(key, obs.shape[0])
    logits = jax.vmap(lambda o: replay(policy, o, ep)[0])(obs)
    if greedy:
        p = jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1])
    else:
        p = jax.nn.softmax(logits, axis=-1)
    return float(0.5 * jnp.abs(p - p[perm]).sum(-1).mean())
