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
from adaptive_therapy.rl.rollout import EpisodeConfig, batch_rollout, replay


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


def _entropy(logits):
    """Exact entropy of a categorical policy, from its logits.

    2026-09-23: replaced a "-logp of the sampled action" proxy. Its expected
    gradient under on-policy sampling is exactly zero (E[grad log pi] = 0), so
    the old bonus never opposed collapse -- and Phase 2's ODE-RNN arm collapsed
    onto open-loop schedules (RESEARCH_LOG L312).
    """
    return -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1)


def train(
    policy,
    thetas,
    y0s,
    key,
    cfg: PPOConfig | None = None,
    ep: EpisodeConfig | None = None,
    rw: RewardConfig | None = None,
):
    """Train `policy` on a cohort. Returns (trained_policy, history).

    Standard recurrent PPO, rewritten 2026-09-23 after Phase 2 (L312). The
    previous loop (a) re-simulated inside the loss and RE-SAMPLED actions with
    the new parameters, so the ratio could compare two different actions; (b)
    left advantages and returns attached to the graph, so the policy loss also
    moved the critic; (c) used a zero-expected-gradient entropy proxy. Now:
    actions, observations, advantages and returns are fixed per update from
    one collection rollout, and the loss replays the stored observations.
    """
    cfg = cfg or PPOConfig()
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig()

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))
    n = thetas.shape[0]

    # Jitted ONCE per train() call. 2026-09-23: called eagerly, the collection
    # rollout recompiled on every update (rollout() builds a fresh scan body
    # each call): ~270 JIT memory mappings per update crossed the kernel's
    # vm.max_map_count=65530 near update 240 -- the LLVM crash of L128/L207.
    @eqx.filter_jit
    def collect(pol, theta, y0, k):
        traj = batch_rollout(pol, theta, y0, k, ep, rw)
        v = traj["value"]
        adv = jax.vmap(lambda rr, vv: _gae(rr, vv, cfg))(traj["reward"] * cfg.reward_scale, v)
        ret = adv + v
        batch = {
            "obs": traj["obs"],
            "action": traj["action"],
            "old_logp": traj["logp"],
            "adv": (adv - adv.mean()) / (adv.std() + 1e-8),
            "ret": ret,
        }
        return jax.lax.stop_gradient(batch), traj

    def loss_fn(pol, batch):
        logits, v = jax.vmap(lambda o: replay(pol, o, ep))(batch["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, batch["action"][..., None], axis=-1)[..., 0]
        ratio = jnp.exp(logp - batch["old_logp"])
        adv = batch["adv"]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = ((v - batch["ret"]) ** 2).mean()
        ent = _entropy(logits).mean()
        return pg + cfg.value_coef * vloss - cfg.entropy_coef * ent, ent

    @eqx.filter_jit
    def update(pol, opt_state, batch):
        (loss, ent), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(pol, batch)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(pol, eqx.is_array))
        return eqx.apply_updates(pol, updates), opt_state, loss, ent

    history = []
    for it in range(cfg.n_updates):
        key, k_sel, k_roll = jax.random.split(key, 3)
        idx = jax.random.choice(k_sel, n, (min(cfg.batch_patients, n),), replace=False)
        batch, old = collect(policy, thetas[idx], y0s[idx], k_roll)

        for _ in range(cfg.n_epochs):
            policy, opt_state, loss, ent = update(policy, opt_state, batch)

        if it % 25 == 0 or it == cfg.n_updates - 1:
            history.append(
                {
                    "update": it,
                    "loss": float(loss),
                    "entropy": float(ent),
                    "mean_reward": float(old["reward"].sum(axis=1).mean()),
                    "mean_dose": float(old["dose"].mean()),
                }
            )

    return policy, history


def evaluate(policy, thetas, y0s, key, ep=None, rw=None, greedy: bool = True):
    """Deterministic evaluation rollout. Returns burden and dose trajectories."""
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig()
    traj = batch_rollout(policy, thetas, y0s, key, ep, rw, greedy=greedy)
    return {
        "burden": jnp.asarray(traj["burden"]),
        "dose": jnp.asarray(traj["dose"]),
        "reward": jnp.asarray(traj["reward"]),
        "obs": jnp.asarray(traj["obs"]),
    }
