"""V14: the S1 comparison under a fourth trainer (evolution strategies on the episode outcome).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-21] V14, before any cell
ran. V12's cohort, seeds (31-40), Strobl ranges, w grid and Tiers 0/1, arm sizes;
the trainer is `adaptive_therapy.rl.es` (antithetic ES, centred ranks, Adam;
fitness = whole-episode TTP-bearing return). Two stages, the second only if the
first passes LAW L1's gate:

  --stage gate      es_mlp and es_ref (S2's full-information MLP)
  --stage compare   adds es_lstm and es_ode; run ONLY after the gate passed

Run: JAX_PLATFORMS=cpu uv run python research/v14_es.py --stage gate
     -> research/outputs/v14_gate.json | v14_es.json, v14_shards/, v14_policies/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

# 2026-10-21: 5 workers x 4 XLA threads, measured 0.43 s/generation on 4 cores
# against 0.9 s with all 20 threads in one process (thread contention).
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
)
N_PARALLEL = 5

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v12_ppo as v12  # noqa: E402  (imports s1_strobl: backend env first)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402
from adaptive_therapy.rl import es  # noqa: E402
from adaptive_therapy.rl.rollout import ORACLE_DIMS  # noqa: E402

s1, w4, g, wf = v12.s1, v12.w4, v12.g, v12.wf
BASE = {"es_mlp": "mlp", "es_lstm": "lstm", "es_ode": "ode_rnn", "es_ref": None}
STAGES = {
    "gate": ("es_mlp", "es_ref"),
    "compare": ("es_mlp", "es_ref", "es_lstm", "es_ode"),
}
CFG = es.ESConfig(n_gens=2000, batch_patients=32)  # the registered defaults; nothing is varied
SHARDS = g.OUT / "v14_shards"
POLICIES = g.OUT / "v14_policies"


def make(arch: str):
    if arch == "es_ref":
        dim = 1 + ORACLE_DIMS["full"]
        return lambda k: MLPPolicy(k, width=441, layer_norm=True, in_dim=dim)
    return w4.ARCHS[BASE[arch]]


def ep_for(arch: str):
    return replace(g.EP, oracle="full") if arch == "es_ref" else g.EP


def cell(arch: str, w: float, seed: int) -> dict:
    k_init, k_train, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
    c = sample_cohort(g.FAM, g.N_TRAIN, seed=seed, ranges=g.RANGES)
    t0 = time.perf_counter()
    policy, hist = es.train(
        make(arch)(k_init),
        jnp.asarray(c.thetas),
        jnp.asarray(c.y0s),
        k_train,
        CFG,
        ep_for(arch),
        RewardConfig(survival_until="progression", burden_penalty=0.0, exposure_penalty=w),
    )
    secs = time.perf_counter() - t0
    w4.save_policy(policy, POLICIES, arch, w, seed)
    return {
        "arch": arch,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        "final_fit": hist[-1]["fit_mean"],
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


def analyse(cells: list[dict], arms: tuple[str, ...]) -> dict:
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells}
    n = len(w4.WEIGHTS) * len(s1.SEEDS_A)
    res = {"prereg": "adaptive_therapy/PREREG.md V14 [2026-10-21]", "vs": {}, "reading": {}}
    res["nonfinite_share"] = {
        a: sum(not valid[(a, w, s)] for w in w4.WEIGHTS for s in s1.SEEDS_A) / n for a in arms
    }
    pairs = [("es_ref", "es_mlp")]
    if "es_lstm" in arms:
        pairs += [("es_lstm", "es_mlp"), ("es_ode", "es_mlp"), ("es_ode", "es_lstm")]
    for tier in s1.TIERS:
        for a, b in pairs:
            res["vs"][f"{a}-{b}_{tier}"] = v12.compare(cells, valid, a, b, tier)
    res["arms"] = {f"{a}_{t}": v12.describe(cells, a, t) for a in arms for t in s1.TIERS}
    res["reading"]["headroom_gate"] = (
        "passed" if res["vs"]["es_ref-es_mlp_tier0"]["ci95"][0] > 0 else "failed"
    )
    if "es_lstm" in arms:
        for a in ("es_lstm", "es_ode"):
            res["reading"][f"{a}_vs_mlp_tier1"] = v12.reading(
                res["vs"][f"{a}-es_mlp_tier1"]["ci95"]
            )
    res["reading"]["valid_arms"] = {a: s <= 0.05 for a, s in res["nonfinite_share"].items()}
    return res


def main(stage: str) -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    arms = STAGES[stage]
    jobs = [(a, w, sd) for a in arms for w in w4.WEIGHTS for sd in s1.SEEDS_A]
    with ThreadPoolExecutor(N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(cells, arms)
    out = "v14_gate.json" if stage == "gate" else "v14_es.json"
    (g.OUT / out).write_text(json.dumps(res, indent=2))
    print(f"nonfinite {res['nonfinite_share']}")
    for k, v in res["vs"].items():
        print(f"  {k:30s} {v12.ci(v)}")
    for k, v in res["arms"].items():
        print(f"  {k:18s} {v}")
    print(f"READING {res['reading']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        (SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(cell(arch, w, seed)))
    else:
        stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "gate"
        sys.exit(main(stage))
