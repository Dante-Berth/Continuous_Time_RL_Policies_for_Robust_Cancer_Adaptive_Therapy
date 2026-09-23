"""PQN -- parallelised Q(lambda) with no replay buffer and no target network.

arxiv-2407.04811 (Alg. 2, p.30; reviewed 1 deg in literature/). Adopted
2026-09-23 as the Phase 2c trainer for BOTH arms after PPO left the ODE-RNN an
open-loop schedule in half its seeds even when corrected (RESEARCH_LOG L312,
L391). The policy's 3 outputs are read as Q-values; its value head is unused.

Why this algorithm and not SAC/TD3 (the owner's first suggestion): there is no
replay buffer, so a recurrent policy needs no stored hidden states or burn-in
-- trajectories are collected with the current network and the loss
back-propagates through time over them (p.7). Exploration is an epsilon floor,
which cannot collapse the way a learned entropy did under PPO.

Two departures from the paper, both forced by recurrence:
  * a minibatch is a set of WHOLE patient episodes, never shuffled
    transitions (Alg. 2 line 9 samples transitions, valid only when the
    network is feedforward);
  * Q(s_t) for the targets is recomputed by `replay` over the stored
    observations, i.e. exactly the values the acting network produced.

Hyperparameter defaults are the paper's recurrent (Craftax RNN) settings,
Table 4 p.36, adopted rather than tuned.
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
class PQNConfig:
    lr: float = 3e-4  # Table 4
    lr_linear_decay: bool = True  # Table 4
    n_updates: int = 300  # same budget as PPO: 300 updates x 32 patients x 52 visits
    batch_patients: int = 32  # parallel environments per update (Table 4 uses 1024)
    n_epochs: int = 4  # Table 4
    n_minibatches: int = 4  # Table 4
    gamma: float = 0.99  # Table 4
    lam: float = 0.5  # Table 4 (RNN); Atari used 0.65
    eps_start: float = 1.0  # Table 4
    eps_finish: float = 0.005  # Table 4
    eps_decay: float = 0.1  # fraction of updates over which eps decays (Table 4)
    max_grad_norm: float = 0.5  # Table 4
    reward_scale: float = 0.01  # as PPO: rewards are ~1/day over 52 visits


def q_lambda_targets(rewards, q, cfg: PQNConfig):
    """Peng's Q(lambda) targets for one fixed-length episode (p.7).

    `rewards` (T,), `q` (T, A) the Q-values at each visit. The episode ends at
    the horizon, so the last target is its reward alone; earlier ones are

        R_t = r_t + gamma * [lam * R_{t+1} + (1 - lam) * max_a Q(s_{t+1}, a)].

    lam=0 is one-step Q-learning; lam=1 the discounted Monte-Carlo return.
    """
    next_max = jnp.max(q[1:], axis=-1)

    def step(r_next, x):
        r, m = x
        r_t = r + cfg.gamma * (cfg.lam * r_next + (1.0 - cfg.lam) * m)
        return r_t, r_t

    last = rewards[-1]
    _, earlier = jax.lax.scan(step, last, (rewards[:-1], next_max), reverse=True)
    return jnp.concatenate([earlier, last[None]])


def episode_minibatches(batch: dict, key, n: int) -> dict:
    """Shuffle PATIENTS, then split into `n` minibatches of whole episodes.

    Leading axis of every leaf is the patient; the time axis is never touched,
    because a recurrent Q-value at visit t depends on visits 0..t.
    """
    n_pat = jax.tree_util.tree_leaves(batch)[0].shape[0]
    perm = jax.random.permutation(key, n_pat)
    return jax.tree_util.tree_map(lambda x: x[perm].reshape(n, n_pat // n, *x.shape[1:]), batch)


def train(
    policy,
    thetas,
    y0s,
    key,
    cfg: PQNConfig | None = None,
    ep: EpisodeConfig | None = None,
    rw: RewardConfig | None = None,
):
    """Train `policy` as a Q-network on a cohort. Returns (trained_policy, history)."""
    cfg = cfg or PQNConfig()
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig()
    if cfg.batch_patients % cfg.n_minibatches:
        raise ValueError("n_minibatches must divide batch_patients")

    n_steps = cfg.n_updates * cfg.n_epochs * cfg.n_minibatches
    lr = optax.linear_schedule(cfg.lr, 0.0, n_steps) if cfg.lr_linear_decay else cfg.lr
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.radam(lr))
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))
    eps_at = optax.linear_schedule(
        cfg.eps_start, cfg.eps_finish, max(1, int(cfg.eps_decay * cfg.n_updates))
    )
    n = thetas.shape[0]

    # Jitted once per train() call -- an eager rollout recompiles every update
    # and leaks JIT mappings until the process dies (RESEARCH_LOG L207).
    @eqx.filter_jit
    def collect(pol, theta, y0, k, eps):
        traj = batch_rollout(pol, theta, y0, k, ep, rw, eps=eps)
        q = jax.vmap(lambda o: replay(pol, o, ep)[0])(traj["obs"])
        targets = jax.vmap(lambda r, qq: q_lambda_targets(r, qq, cfg))(
            traj["reward"] * cfg.reward_scale, q
        )
        batch = {"obs": traj["obs"], "action": traj["action"], "target": targets}
        return jax.lax.stop_gradient(batch), traj

    def loss_fn(pol, mb):
        q = jax.vmap(lambda o: replay(pol, o, ep)[0])(mb["obs"])
        q_a = jnp.take_along_axis(q, mb["action"][..., None], axis=-1)[..., 0]
        return 0.5 * jnp.mean((q_a - mb["target"]) ** 2)

    @eqx.filter_jit
    def learn(pol, opt_state, batch, k):
        params, static = eqx.partition(pol, eqx.is_array)

        def minibatch_step(carry, mb):
            params, opt_state = carry
            p = eqx.combine(params, static)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(p, mb)
            updates, opt_state = opt.update(grads, opt_state, eqx.filter(p, eqx.is_array))
            params = eqx.filter(eqx.apply_updates(p, updates), eqx.is_array)
            return (params, opt_state), loss

        def epoch(carry, k_ep):
            mbs = episode_minibatches(batch, k_ep, cfg.n_minibatches)
            return jax.lax.scan(minibatch_step, carry, mbs)

        (params, opt_state), losses = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(k, cfg.n_epochs)
        )
        return eqx.combine(params, static), opt_state, losses.mean()

    history = []
    for it in range(cfg.n_updates):
        key, k_sel, k_roll, k_learn = jax.random.split(key, 4)
        idx = jax.random.choice(k_sel, n, (min(cfg.batch_patients, n),), replace=False)
        eps = eps_at(it)
        batch, traj = collect(policy, thetas[idx], y0s[idx], k_roll, eps)
        policy, opt_state, loss = learn(policy, opt_state, batch, k_learn)

        if it % 25 == 0 or it == cfg.n_updates - 1:
            history.append(
                {
                    "update": it,
                    "loss": float(loss),
                    "eps": float(eps),
                    "mean_reward": float(traj["reward"].sum(axis=1).mean()),
                    "mean_dose": float(traj["dose"].mean()),
                }
            )

    return policy, history
