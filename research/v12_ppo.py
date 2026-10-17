"""V12: the S1 comparison under a second trainer (PPO instead of PQN).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-18] V12, before any cell
ran. S1 Part A's cohort, seeds (31-40), Strobl ranges, w grid and Tiers 0/1,
trained at p = 0, but the Q-network/policy is trained by the repo's recurrent
PPO (`adaptive_therapy.rl.ppo`, rewritten 2026-09-23 after L312), 600 updates.
The policy outputs are read as logits; all arms are compared with each other
inside this run (no PQN shards are reused).

  ppo_mlp / ppo_lstm / ppo_ode   the three S1 arms, S1's sizes
  ppo_ref                        the S2 full-information MLP (oracle = "full"),
                                 LAW L1's headroom gate for a new trainer

Run: JAX_PLATFORMS=cpu .venv/bin/python research/v12_ppo.py
     -> research/outputs/v12_ppo.json, v12_shards/, v12_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets Strobl ranges; backend env first)
from v1_review import median_ci  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402
from adaptive_therapy.rl import ppo  # noqa: E402
from adaptive_therapy.rl.rollout import ORACLE_DIMS  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402

w4, g, wf = s1.w4, s1.g, s1.wf
BASE = {"ppo_mlp": "mlp", "ppo_lstm": "lstm", "ppo_ode": "ode_rnn", "ppo_ref": None}
CFG = ppo.PPOConfig(n_updates=600)
SHARDS = g.OUT / "v12_shards"
POLICIES = g.OUT / "v12_policies"
GATE = 10.0  # days; the program's gate, as registered


def make(arch: str):
    if arch == "ppo_ref":
        dim = 1 + ORACLE_DIMS["full"]
        return lambda k: MLPPolicy(k, width=441, layer_norm=True, in_dim=dim)
    return w4.ARCHS[BASE[arch]]


def ep_for(arch: str):
    return replace(g.EP, oracle="full") if arch == "ppo_ref" else g.EP


def cell(arch: str, w: float, seed: int) -> dict:
    k_init, k_train, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
    c = sample_cohort(g.FAM, g.N_TRAIN, seed=seed, ranges=g.RANGES)
    t0 = time.perf_counter()
    policy, hist = ppo.train(
        make(arch)(k_init),
        jnp.asarray(c.thetas),
        jnp.asarray(c.y0s),
        k_train,
        CFG,
        ep_for(arch),
        RewardConfig(exposure_penalty=w),
    )
    secs = time.perf_counter() - t0
    w4.save_policy(policy, POLICIES, arch, w, seed)
    return {
        "arch": arch,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        "final_entropy": hist[-1]["entropy"],
        **s1.tiers_eval(policy, seed, w, ep=ep_for(arch)),
    }


def spawn(arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {arch} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def compare(cells, valid, a, b, tier) -> dict:
    d = w4.paired(cells, valid, a, b, "level", s1.SEEDS_A, (tier,))
    return {**w4.delta_ci(d, (tier,)), "median": median_ci(d)}


def describe(cells, arch, tier) -> dict:
    cs = [c for c in cells if c["arch"] == arch]
    pool = lambda k: np.concatenate([np.asarray(c[tier][k]) for c in cs])  # noqa: E731
    q = lambda x: [float(v) for v in np.percentile(x, [25, 50, 75])]  # noqa: E731
    return {
        "ttp_capped_q25_50_75": q(pool("ttp_capped")),
        "exposure_q25_50_75": q(pool("exposure")),
        "burden_integral_q25_50_75": q(pool("burden_integral")),
        "closed_loop_mean": float(np.mean([c[f"closed_loop_{tier}"] for c in cs])),
        "open_loop_cells": int(sum(c[f"closed_loop_{tier}"] < 0.05 for c in cs)),
    }


def reading(ci95) -> str:
    return "edge" if ci95[0] > 0 else "deficit" if ci95[1] < 0 else "ties"


def analyse(cells: list[dict]) -> dict:
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells}
    n = len(w4.WEIGHTS) * len(s1.SEEDS_A)
    res = {"prereg": "adaptive_therapy/PREREG.md V12 [2026-10-18]", "vs": {}, "reading": {}}
    res["nonfinite_share"] = {
        a: sum(not valid[(a, w, s)] for w in w4.WEIGHTS for s in s1.SEEDS_A) / n for a in BASE
    }
    for tier in s1.TIERS:
        for a in ("ppo_lstm", "ppo_ode", "ppo_ref"):
            res["vs"][f"{a}-ppo_mlp_{tier}"] = compare(cells, valid, a, "ppo_mlp", tier)
        res["vs"][f"ppo_ode-ppo_lstm_{tier}"] = compare(cells, valid, "ppo_ode", "ppo_lstm", tier)
    res["arms"] = {f"{a}_{t}": describe(cells, a, t) for a in BASE for t in s1.TIERS}
    res["reading"]["headroom_gate"] = (
        "passed" if res["vs"]["ppo_ref-ppo_mlp_tier0"]["ci95"][0] > 0 else "failed"
    )
    for a in ("ppo_lstm", "ppo_ode"):
        res["reading"][f"{a}_vs_mlp_tier1"] = reading(res["vs"][f"{a}-ppo_mlp_tier1"]["ci95"])
    res["reading"]["valid_arms"] = {a: s <= 0.05 for a, s in res["nonfinite_share"].items()}
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, sd) for a in BASE for w in w4.WEIGHTS for sd in s1.SEEDS_A]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(cells)
    (g.OUT / "v12_ppo.json").write_text(json.dumps(res, indent=2))
    print(f"nonfinite {res['nonfinite_share']}")
    for k, v in res["vs"].items():
        print(f"  {k:28s} {ci(v)}")
    for k, v in res["arms"].items():
        print(f"  {k:18s} {v}")
    print(f"READING {res['reading']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        (SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(cell(arch, w, seed)))
    else:
        sys.exit(main())
