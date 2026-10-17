"""W10: does the PQN trainer used for every arm let a recurrent net use memory when memory pays?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-17] W10, before any cell
ran (the 20-update smoke run only timed the code). POPGym tasks in JAX
(popjaxrl, Lu et al. 2023, not vendored: no licence; clone to
research/outputs/popjaxrl or set POPJAXRL_DIR). Same PQN update as the cancer
runs (`adaptive_therapy.rl.pqn`: Q(lambda) targets, whole-episode minibatches,
Table-4 recurrent defaults); only the environment differs.

  mlp   memoryless Q-net (2 hidden layers, LayerNorm readout)
  lstm  LSTM Q-net (LayerNorm readout), parameter count within 5% of the MLP

Run one cell: JAX_PLATFORMS=cpu uv run python research/w10_headroom.py cell TASK ARCH SEED
     all:     JAX_PLATFORMS=cpu uv run python research/w10_headroom.py all
     -> research/outputs/w10_headroom.json, w10_shards/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("POPJAXRL_DIR", str(ROOT / "outputs" / "popjaxrl")))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from envs.registration import make  # noqa: E402

from adaptive_therapy.rl.pqn import PQNConfig, episode_minibatches, q_lambda_targets  # noqa: E402

OUT = ROOT / "outputs"
SHARDS = OUT / "w10_shards"
TASKS = ("RepeatPreviousEasy", "RepeatFirstEasy", "CountRecallEasy")
ARCHS = ("mlp", "lstm")
SEEDS = tuple(range(101, 111))
N_ENVS = 64  # parallel episodes per update (the cancer runs use 32 patients)
N_EVAL = 512  # greedy evaluation episodes per cell
REWARD_SCALE = 10.0  # per-step rewards here are ~1/48; the cancer runs scale ~1/day by 0.01
CFG = PQNConfig(n_updates=2000, batch_patients=N_ENVS, reward_scale=REWARD_SCALE)


class MLPQ(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(self, key, obs_dim, n_act, width=128):
        k1, k2, k3 = jax.random.split(key, 3)
        self.l1, self.l2 = (
            eqx.nn.Linear(obs_dim, width, key=k1),
            eqx.nn.Linear(width, width, key=k2),
        )
        self.norm, self.head = eqx.nn.LayerNorm(width), eqx.nn.Linear(width, n_act, key=k3)

    def initial_state(self):
        return jnp.zeros((1,))

    def __call__(self, obs, state):
        h = jax.nn.relu(self.l2(jax.nn.relu(self.l1(obs))))
        return self.head(self.norm(h)), state


class LSTMQ(eqx.Module):
    cell: eqx.nn.LSTMCell
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    hidden: int = eqx.field(static=True)

    def __init__(self, key, obs_dim, n_act, hidden=64):
        k1, k2 = jax.random.split(key)
        self.hidden = hidden
        self.cell = eqx.nn.LSTMCell(obs_dim, hidden, key=k1)
        self.norm, self.head = eqx.nn.LayerNorm(hidden), eqx.nn.Linear(hidden, n_act, key=k2)

    def initial_state(self):
        return jnp.zeros((2 * self.hidden,))

    def __call__(self, obs, state):
        h, c = self.cell(obs, (state[: self.hidden], state[self.hidden :]))
        return self.head(self.norm(h)), jnp.concatenate([h, c])


def n_params(net) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(net, eqx.is_array))))


def build(task: str, arch: str, key):
    env, params = make(task)
    obs_dim = env.observation_space(params).shape[0]
    n_act = env.action_space(params).n
    net = (MLPQ if arch == "mlp" else LSTMQ)(key, obs_dim, n_act)
    return env, params, net, env.decksize * env.num_decks


def make_fns(env, params, horizon):
    def rollout(net, key, eps, n):
        k_reset, k_run = jax.random.split(key)
        obs, st = jax.vmap(lambda k: env.reset(k, params))(jax.random.split(k_reset, n))
        h0 = jnp.broadcast_to(net.initial_state(), (n, *net.initial_state().shape))

        def step(carry, k):
            obs, st, h = carry
            ka, ke, ks = jax.random.split(k, 3)
            q, h2 = jax.vmap(net)(obs, h)
            greedy = jnp.argmax(q, -1)
            rand = jax.random.randint(ka, (n,), 0, q.shape[-1])
            act = jnp.where(jax.random.uniform(ke, (n,)) < eps, rand, greedy)
            obs2, st2, r, _, _ = jax.vmap(lambda k, s, a: env.step(k, s, a, params))(
                jax.random.split(ks, n), st, act
            )
            return (obs2, st2, h2), {"obs": obs, "action": act, "reward": r}

        _, tr = jax.lax.scan(step, (obs, st, h0), jax.random.split(k_run, horizon))
        return jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), tr)  # (n, T, ...)

    def replay(net, obs):
        def step(h, o):
            q, h2 = net(o, h)
            return h2, q

        return jax.lax.scan(step, net.initial_state(), obs)[1]

    return rollout, replay


def train_cell(task: str, arch: str, seed: int, cfg: PQNConfig = CFG):
    key = jax.random.PRNGKey(seed)
    key, k_net = jax.random.split(key)
    env, params, net, horizon = build(task, arch, k_net)
    rollout, replay = make_fns(env, params, horizon)
    n_steps = cfg.n_updates * cfg.n_epochs * cfg.n_minibatches
    opt = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.radam(optax.linear_schedule(cfg.lr, 0.0, n_steps)),
    )
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    eps_at = optax.linear_schedule(
        cfg.eps_start, cfg.eps_finish, max(1, int(cfg.eps_decay * cfg.n_updates))
    )

    @eqx.filter_jit
    def collect(net, k, eps):
        tr = rollout(net, k, eps, cfg.batch_patients)
        q = jax.vmap(lambda o: replay(net, o))(tr["obs"])
        tgt = jax.vmap(lambda r, qq: q_lambda_targets(r * cfg.reward_scale, qq, cfg))(
            tr["reward"], q
        )
        return jax.lax.stop_gradient({"obs": tr["obs"], "action": tr["action"], "target": tgt}), tr[
            "reward"
        ]

    def loss_fn(net, mb):
        q = jax.vmap(lambda o: replay(net, o))(mb["obs"])
        q_a = jnp.take_along_axis(q, mb["action"][..., None], -1)[..., 0]
        return 0.5 * jnp.mean((q_a - mb["target"]) ** 2)

    @eqx.filter_jit
    def learn(net, opt_state, batch, k):
        params_, static = eqx.partition(net, eqx.is_array)

        def mb_step(carry, mb):
            p, os_ = carry
            m = eqx.combine(p, static)
            loss, g = eqx.filter_value_and_grad(loss_fn)(m, mb)
            up, os_ = opt.update(g, os_, eqx.filter(m, eqx.is_array))
            return (eqx.filter(eqx.apply_updates(m, up), eqx.is_array), os_), loss

        def epoch(carry, k_ep):
            return jax.lax.scan(mb_step, carry, episode_minibatches(batch, k_ep, cfg.n_minibatches))

        (p, opt_state), losses = jax.lax.scan(
            epoch, (params_, opt_state), jax.random.split(k, cfg.n_epochs)
        )
        return eqx.combine(p, static), opt_state, losses.mean()

    curve = []
    for it in range(cfg.n_updates):
        key, k_roll, k_learn = jax.random.split(key, 3)
        batch, rew = collect(net, k_roll, eps_at(it))
        net, opt_state, _ = learn(net, opt_state, batch, k_learn)
        if it % 100 == 0 or it == cfg.n_updates - 1:
            curve.append({"update": it, "train_return": float(rew.sum(1).mean())})

    @eqx.filter_jit
    def evaluate(net, k):
        return rollout(net, k, 0.0, N_EVAL)["reward"].sum(1)

    ret = np.asarray(evaluate(net, jax.random.PRNGKey(10_000 + seed)))
    return {
        "task": task,
        "arch": arch,
        "seed": seed,
        "n_params": n_params(net),
        "eval_return_mean": float(ret.mean()),
        "eval_return_sd": float(ret.std()),
        "curve": curve,
    }


def cell_path(task, arch, seed) -> Path:
    return SHARDS / f"{task}_{arch}_{seed}.json"


def run_all(workers: int = 18):
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [
        (t, a, s) for t in TASKS for a in ARCHS for s in SEEDS if not cell_path(t, a, s).exists()
    ]

    def one(j):
        subprocess.run([sys.executable, __file__, "cell", *map(str, j)], check=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    rows = [json.loads(p.read_text()) for p in sorted(SHARDS.glob("*.json"))]
    (OUT / "w10_headroom.json").write_text(json.dumps(rows, indent=1))


def boot_ci(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    m = rng.choice(d, (n, len(d))).mean(1)
    return float(d.mean()), *np.percentile(m, [2.5, 97.5]).tolist()


if __name__ == "__main__":
    if sys.argv[1] == "cell":
        task, arch, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        n_up = int(os.environ.get("W10_UPDATES", CFG.n_updates))
        import dataclasses

        res = train_cell(task, arch, seed, dataclasses.replace(CFG, n_updates=n_up))
        if "W10_UPDATES" not in os.environ:
            SHARDS.mkdir(parents=True, exist_ok=True)
            cell_path(task, arch, seed).write_text(json.dumps(res))
        print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    else:
        run_all()
