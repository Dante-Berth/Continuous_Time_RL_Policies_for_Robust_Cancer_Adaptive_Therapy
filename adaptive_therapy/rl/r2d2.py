"""Episodic-replay recurrent Q-learning -- R2D2's recipe on whole episodes.

PREREG V13 [2026-10-21]. The third trainer, after PPO (V12, uninformative: its
full-information reference did not beat the MLP) and PQN. It differs from PQN in
exactly the pieces R2D2 adds to a recurrent DQN: a replay buffer (each episode
is reused many times), a slowly moving target network, and n-step double-Q
targets. R2D2's stored hidden state and burn-in exist because it replays
sub-sequences of long episodes; here an episode is 52 visits from a known
initial state, so the buffer stores WHOLE episodes and `replay` re-runs the
network from its zero state -- there is no stale state to repair.

The policy's 3 outputs are read as Q-values; its value head is unused (as PQN).
Regular visits only: a missed visit is not a decision, and the semi-MDP
bookkeeping lives in pqn.py, which V13 does not touch.

Hyperparameters are R2D2's (Kapturowski et al. 2019) where the paper gives one,
PQN's elsewhere -- adopted, not tuned (PREREG V13 declares them).
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
class R2D2Config:
    lr: float = 3e-4  # PQN's; R2D2 used 1e-4 with far more data
    lr_linear_decay: bool = True  # as PQN
    n_updates: int = 600  # as V12's PPO: 600 x 32 episodes of interaction
    batch_patients: int = 32  # episodes collected per update
    grads_per_update: int = 4  # replay ratio: 4 x 32 sampled episodes per 32 collected
    batch_episodes: int = 32  # episodes per gradient step
    capacity: int = 2048  # buffer size, in episodes (about 64 updates of data)
    n_step: int = 5  # R2D2 uses 5
    gamma: float = 0.99
    tau: float = 0.005  # target network: Polyak averaging per gradient step
    eps_start: float = 1.0
    eps_finish: float = 0.005
    eps_decay: float = 0.1  # fraction of updates over which eps decays (as PQN)
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01  # as PQN and PPO


def n_step_double_q_targets(rewards, q_online, q_target, cfg: R2D2Config):
    """n-step double-Q targets for one fixed-length episode.

    `rewards` (T,), `q_online` / `q_target` (T, A). With n = cfg.n_step,

        y_t = sum_{k<n} gamma^k r_{t+k}  +  gamma^n Q_tgt(s_{t+n}, argmax_a Q_on(s_{t+n}, a)),

    the bootstrap term absent when t + n >= T: the episode ends at the horizon,
    so the tail of the return is the rewards that remain and nothing else.
    """
    t = rewards.shape[0]
    n = cfg.n_step
    padded = jnp.concatenate([rewards, jnp.zeros(n)])
    ret = sum(cfg.gamma**k * padded[k : k + t] for k in range(n))
    nxt = jnp.arange(t) + n
    inside = nxt < t
    nxt = jnp.minimum(nxt, t - 1)
    best = jnp.argmax(q_online[nxt], axis=-1)
    boot = jnp.take_along_axis(q_target[nxt], best[:, None], axis=-1)[:, 0]
    return ret + jnp.where(inside, cfg.gamma**n * boot, 0.0)


def buffer_insert(buf: dict, new: dict, ptr: int, capacity: int):
    """Write `new` (B episodes) into the ring buffer at `ptr`; returns (buf, ptr')."""
    b = jax.tree_util.tree_leaves(new)[0].shape[0]
    pos = (ptr + jnp.arange(b)) % capacity
    return jax.tree_util.tree_map(lambda x, y: x.at[pos].set(y), buf, new), (ptr + b) % capacity


def train(
    policy,
    thetas,
    y0s,
    key,
    cfg: R2D2Config | None = None,
    ep: EpisodeConfig | None = None,
    rw: RewardConfig | None = None,
):
    """Train `policy` as a Q-network on a cohort. Returns (trained_policy, history)."""
    cfg = cfg or R2D2Config()
    ep = ep or EpisodeConfig()
    rw = rw or RewardConfig()
    if ep.drop_rate or ep.schedule is not None or ep.schedule_pool is not None:
        raise ValueError("r2d2 trains on regular visits only (every visit a decision)")

    n_grad = cfg.n_updates * cfg.grads_per_update
    lr = optax.linear_schedule(cfg.lr, 0.0, n_grad) if cfg.lr_linear_decay else cfg.lr
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr))
    params0, static = eqx.partition(policy, eqx.is_array)
    opt_state = opt.init(params0)
    eps_at = optax.linear_schedule(
        cfg.eps_start, cfg.eps_finish, max(1, int(cfg.eps_decay * cfg.n_updates))
    )
    n = thetas.shape[0]

    def q_of(params, obs):
        pol = eqx.combine(params, static)
        return jax.vmap(lambda o: replay(pol, o, ep)[0])(obs)

    # Jitted once per train() call -- an eager rollout recompiles every update
    # and leaks JIT mappings until the process dies (RESEARCH_LOG L207).
    @eqx.filter_jit
    def collect(pol, theta, y0, k, eps):
        traj = batch_rollout(pol, theta, y0, k, ep, rw, eps=eps)
        episodes = {
            "obs": traj["obs"],
            "action": traj["action"],
            "reward": traj["reward"] * cfg.reward_scale,
        }
        return jax.lax.stop_gradient(episodes), traj

    @eqx.filter_jit
    def insert(buf, new, ptr):
        return buffer_insert(buf, new, ptr, cfg.capacity)

    def loss_fn(params, target_params, mb):
        q_on = q_of(params, mb["obs"])
        q_tg = jax.lax.stop_gradient(q_of(target_params, mb["obs"]))
        y = jax.vmap(lambda r, a, b: n_step_double_q_targets(r, a, b, cfg))(
            mb["reward"], jax.lax.stop_gradient(q_on), q_tg
        )
        q_a = jnp.take_along_axis(q_on, mb["action"][..., None], axis=-1)[..., 0]
        return optax.huber_loss(q_a, y).mean()

    @eqx.filter_jit
    def learn(params, target_params, opt_state, buf, size, k):
        def grad_step(carry, k_i):
            params, target_params, opt_state = carry
            idx = jax.random.randint(k_i, (cfg.batch_episodes,), 0, size)
            mb = jax.tree_util.tree_map(lambda x: x[idx], buf)
            loss, grads = jax.value_and_grad(loss_fn)(params, target_params, mb)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            target_params = optax.incremental_update(params, target_params, cfg.tau)
            return (params, target_params, opt_state), loss

        carry, losses = jax.lax.scan(
            grad_step,
            (params, target_params, opt_state),
            jax.random.split(k, cfg.grads_per_update),
        )
        return (*carry, losses.mean())

    params, target_params = params0, params0
    buf, ptr, filled = None, jnp.asarray(0), 0
    history = []
    for it in range(cfg.n_updates):
        key, k_sel, k_roll, k_learn = jax.random.split(key, 4)
        idx = jax.random.choice(k_sel, n, (min(cfg.batch_patients, n),), replace=False)
        eps = eps_at(it)
        new, traj = collect(eqx.combine(params, static), thetas[idx], y0s[idx], k_roll, eps)
        if buf is None:
            buf = jax.tree_util.tree_map(
                lambda x: jnp.zeros((cfg.capacity, *x.shape[1:]), x.dtype), new
            )
        buf, ptr = insert(buf, new, ptr)
        filled = min(cfg.capacity, filled + cfg.batch_patients)
        params, target_params, opt_state, loss = learn(
            params, target_params, opt_state, buf, jnp.asarray(filled), k_learn
        )

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

    return eqx.combine(params, static), history
