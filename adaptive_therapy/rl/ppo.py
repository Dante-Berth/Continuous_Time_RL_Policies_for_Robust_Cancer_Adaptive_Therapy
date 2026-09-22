"""Minimal PPO for equinox policies.

PROJECT.md 12: purejaxrl assumes Flax-style modules, so a full-equinox stack
means writing the (short) loop here rather than importing one. Kept deliberately
small -- the research question is the ARCHITECTURE, and every knob added here is
a confound that has to be held fixed across arms anyway.

The same loop trains every architecture. The policy is a black box to it beyond
`initial_state`, `__call__` and `value`.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.rl.rollout import EpisodeConfig, rollout


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    n_updates: int = 300
    batch_patients: int = 32
    n_epochs: int = 4
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01  # rewards are ~1/day over 52 visits


def _gae(rewards, values, cfg: PPOConfig):
    """Generalised advantage estimate over one episode (no bootstrapping:
    episodes are fixed-length and end at the horizon)."""

    def step(carry, x):
        adv, next_v = carry
        r, v = x
        delta = r + cfg.gamma * next_v - v
        adv = delta + cfg.gamma * cfg.gae_lambda * adv
        return (adv, v), adv

    (_, _), advs = jax.lax.scan(
        step,
        (jnp.array(0.0), jnp.array(0.0)),
        (rewards, values),
        reverse=True,
    )
    return advs


def _batch_rollout(policy, thetas, y0s, key, ep, rw, greedy=False):
    keys = jax.random.split(key, thetas.shape[0])
    return jax.vmap(lambda th, y, k: rollout(policy, th, y, k, ep, rw, greedy=greedy))(
        thetas, y0s, keys
    )


def train(
    policy,
    thetas,
    y0s,
    key,
    cfg: PPOConfig | None = None,
    ep: EpisodeConfig | None = None,
    rw: RewardConfig | None = None,
):
    """Train `policy` on a cohort. Returns (trained_policy, history)."""
    cfg = cfg or PPOConfig()
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig()

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))
    n = thetas.shape[0]

    def loss_fn(pol, batch):
        traj = _batch_rollout(pol, batch["theta"], batch["y0"], batch["key"], ep, rw)
        r = traj["reward"] * cfg.reward_scale
        v = traj["value"]

        adv = jax.vmap(lambda rr, vv: _gae(rr, vv, cfg))(r, v)
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ratio = jnp.exp(traj["logp"] - batch["old_logp"])
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = ((v - ret) ** 2).mean()
        # Entropy from the realised log-probs: a cheap proxy that needs no
        # second forward pass, and enough to stop premature collapse onto a
        # single dose -- which is the failure mode here, since constant max
        # dosing is a strong TTP attractor.
        ent = -traj["logp"].mean()
        return pg + cfg.value_coef * vloss - cfg.entropy_coef * ent

    @eqx.filter_jit
    def update(pol, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(pol, batch)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(pol, eqx.is_array))
        return eqx.apply_updates(pol, updates), opt_state, loss

    history = []
    for it in range(cfg.n_updates):
        key, k_sel, k_roll = jax.random.split(key, 3)
        idx = jax.random.choice(k_sel, n, (min(cfg.batch_patients, n),), replace=False)
        batch = {"theta": thetas[idx], "y0": y0s[idx], "key": k_roll}

        old = _batch_rollout(policy, batch["theta"], batch["y0"], k_roll, ep, rw)
        batch["old_logp"] = jax.lax.stop_gradient(old["logp"])

        for _ in range(cfg.n_epochs):
            policy, opt_state, loss = update(policy, opt_state, batch)

        if it % 25 == 0 or it == cfg.n_updates - 1:
            history.append(
                {
                    "update": it,
                    "loss": float(loss),
                    "mean_reward": float(old["reward"].sum(axis=1).mean()),
                    "mean_dose": float(old["dose"].mean()),
                }
            )

    return policy, history


def evaluate(policy, thetas, y0s, key, ep=None, rw=None, greedy: bool = True):
    """Deterministic evaluation rollout. Returns burden and dose trajectories."""
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig()
    traj = _batch_rollout(policy, thetas, y0s, key, ep, rw, greedy=greedy)
    return {
        "burden": jnp.asarray(traj["burden"]),
        "dose": jnp.asarray(traj["dose"]),
        "reward": jnp.asarray(traj["reward"]),
    }
