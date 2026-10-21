"""Evolution strategies on the episode outcome -- a trainer with no TD learning.

PREREG V14 [2026-10-21]. The fourth trainer, after PPO, PQN and replay Q-learning
(all three fail LAW L1's headroom gate). It shares none of their machinery: no
value function, no bootstrapping, no credit assignment through time, no action
exploration. The policy acts greedily; the parameters are perturbed instead
(OpenAI-ES, Salimans et al. 2017: antithetic Gaussian perturbations, centred
ranks, Adam on the estimated gradient). Fitness is the whole-episode return of a
TTP-bearing reward (`survival_until="progression"`, no burden term -- V2's), so
it is the trained objective's own episode total, scored on the same cohort
minibatch for every member (common random numbers: members differ only in theta).

Recurrent arms need nothing special: a member is a flat parameter vector and an
episode is a `batch_rollout`.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.rl.rollout import EpisodeConfig, batch_rollout


@dataclass(frozen=True)
class ESConfig:
    n_gens: int = 1500
    pop: int = 64  # members per generation, in antithetic pairs
    batch_patients: int = 16  # episodes per member; same patients for every member
    sigma: float = 0.05  # perturbation scale (Salimans: 0.02, on far larger nets)
    lr: float = 0.02  # Adam on the ES gradient
    lr_linear_decay: bool = True
    weight_decay: float = 0.0


def centred_ranks(f: jnp.ndarray) -> jnp.ndarray:
    """Fitness -> ranks scaled to [-0.5, 0.5]; invariant to the reward's scale and outliers."""
    ranks = jnp.argsort(jnp.argsort(f)).astype(f.dtype)
    return ranks / (f.shape[0] - 1) - 0.5


def es_gradient(eps: jnp.ndarray, fitness: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Ascent direction from antithetic noise `eps` (P/2, d) and fitness (P,).

    Members are [theta + sigma*eps ; theta - sigma*eps]; the estimate is
    sum_i (F+_i - F-_i) eps_i / (P sigma) on centred ranks.
    """
    half = eps.shape[0]
    shaped = centred_ranks(fitness)
    return ((shaped[:half] - shaped[half:]) @ eps) / (fitness.shape[0] * sigma)


def train(
    policy,
    thetas,
    y0s,
    key,
    cfg: ESConfig | None = None,
    ep: EpisodeConfig | None = None,
    rw: RewardConfig | None = None,
):
    """Train `policy` by ES on a cohort. Returns (trained_policy, history)."""
    cfg = cfg or ESConfig()
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig(survival_until="progression", burden_penalty=0.0)
    if cfg.pop % 2:
        raise ValueError("pop must be even (antithetic pairs)")
    if ep.drop_rate or ep.schedule is not None or ep.schedule_pool is not None:
        raise ValueError("es trains on regular visits only (every visit a decision)")

    params, static = eqx.partition(policy, eqx.is_inexact_array)
    flat0, unravel = ravel_pytree(params)
    d, half = flat0.shape[0], cfg.pop // 2
    lr = optax.linear_schedule(cfg.lr, 0.0, cfg.n_gens) if cfg.lr_linear_decay else cfg.lr
    opt = optax.adam(lr)  # minimises: fed the negated ascent direction

    def episode_return(flat, th, y0, k):
        pol = eqx.combine(unravel(flat), static)
        traj = batch_rollout(pol, th, y0, k, ep, rw, greedy=True)
        return traj["reward"].sum(axis=1).mean()  # mean over the minibatch

    @eqx.filter_jit
    def generation(flat, opt_state, th, y0, k_eps, k_roll):
        eps = jax.random.normal(k_eps, (half, d))
        members = jnp.concatenate([flat + cfg.sigma * eps, flat - cfg.sigma * eps])
        fitness = jax.vmap(lambda m: episode_return(m, th, y0, k_roll))(members)
        g = es_gradient(eps, fitness, cfg.sigma) - cfg.weight_decay * flat
        updates, opt_state = opt.update(-g, opt_state, flat)
        return optax.apply_updates(flat, updates), opt_state, fitness

    n = thetas.shape[0]
    flat, opt_state = flat0, opt.init(flat0)
    history = []
    for it in range(cfg.n_gens):
        key, k_sel, k_eps, k_roll = jax.random.split(key, 4)
        idx = jax.random.choice(k_sel, n, (min(cfg.batch_patients, n),), replace=False)
        flat, opt_state, fitness = generation(flat, opt_state, thetas[idx], y0s[idx], k_eps, k_roll)
        if it % 25 == 0 or it == cfg.n_gens - 1:
            history.append(
                {
                    "gen": it,
                    "fit_mean": float(fitness.mean()),
                    "fit_max": float(fitness.max()),
                    "finite": bool(jnp.isfinite(fitness).all()),
                }
            )
    return eqx.combine(unravel(flat), static), history
