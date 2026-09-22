"""Closed-loop episode rollout: policy <-> simulator, burden-only.

One episode = a virtual patient observed at monitoring visits. At each visit the
policy sees ONLY the noisy scalar burden (PROJECT.md 7), picks a dose, and the
simulator integrates to the next visit under that dose held constant -- boluses
with exponential PK, not continuous control.

The loop is IDENTICAL for every architecture. Everything that differs between
arms lives inside the policy object, which is what makes the Phase 2/3 ablation
an architecture comparison rather than a comparison of two systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from adaptive_therapy.envs.base import ModelId, PKConfig, SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_one_patient
from adaptive_therapy.envs.pk import administer, decay, effect
from adaptive_therapy.envs.reward import RewardConfig, step_reward
from adaptive_therapy.policies.base import DOSE_LEVELS, normalise_obs


@dataclass(frozen=True)
class EpisodeConfig:
    n_visits: int = 52
    visit_interval: float = 7.0  # days
    obs_noise: float = 0.05
    model_id: ModelId = "lotka_volterra"


def rollout(
    policy,
    theta: jnp.ndarray,
    y0: jnp.ndarray,
    key,
    ep: EpisodeConfig,
    rw: RewardConfig,
    pk: PKConfig | None = None,
    greedy: bool = False,
):
    """Run one patient. Returns per-visit trajectories for training and eval."""
    pk = pk or PKConfig()
    cfg = SolverConfig(t0=0.0, t1=ep.visit_interval, dt0=0.05, n_save=2)

    baseline = observe(y0, ep.model_id)

    def step(carry, _):
        state, y, conc, h, nadir, k = carry
        k, k_obs, k_act = jax.random.split(k, 3)

        true_b = observe(y, ep.model_id)
        noisy = true_b * jnp.exp(jax.random.normal(k_obs) * ep.obs_noise)
        obs = normalise_obs(noisy, baseline)

        logits, h2 = policy(obs, h, ep.visit_interval)
        action = jnp.where(
            greedy,
            jnp.argmax(logits),
            jax.random.categorical(k_act, logits),
        )
        logp = jax.nn.log_softmax(logits)[action]
        dose = DOSE_LEVELS[action]

        conc2 = administer(conc, dose, pk)
        u = effect(conc2)
        y2 = solve_one_patient(ep.model_id, theta, y, u, cfg)[-1]
        conc3 = decay(conc2, ep.visit_interval, pk)

        nadir2 = jnp.minimum(nadir, true_b)
        r = step_reward(true_b, nadir2, dose, ep.visit_interval, rw)

        out = {
            "burden": true_b,
            "dose": dose,
            "reward": r,
            "logp": logp,
            "action": action,
            "obs": obs,
            "value": policy.value(obs, h),
        }
        return (state, y2, conc3, h2, nadir2, k), out

    init = (None, y0, jnp.array(0.0), policy.initial_state(), observe(y0, ep.model_id), key)
    _, traj = jax.lax.scan(step, init, None, length=ep.n_visits)
    return traj
