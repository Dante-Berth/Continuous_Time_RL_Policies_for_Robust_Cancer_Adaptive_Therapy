"""S3: does PQN at its published parallelism let the reference beat the MLP?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-29] S3, before any S3
cell ran. LAW L1's headroom gate (RESEARCH_LOG L804): a new trainer or budget
is admissible only if, under it, S2's full-information reference beats the
burden-only MLP. No continuous-time arm runs here.

  Trainer   PQNConfig(batch_patients=1024) from a 2048-patient training
            cohort per seed (was 32 from 200); all else as S2.
  Arms      MLP and O_full, both retrained; seeds {31..40}, Strobl ranges.
  Primary   Delta(O_full, MLP) at Tier 0 (seed+patient CI) and R = mean over
            cells of the reference's minus the MLP's training return on its
            own training cohort. lo > +10 d => headroom; else R < 0 =>
            still trainer-limited; else hi < 0 => reward-limited; else
            hi < +10 d => no headroom; else inconclusive.

Run: JAX_PLATFORMS=cpu uv run python research/s3_budget.py
     -> research/outputs/s3_budget.{json,log}, s3_shards/, s3_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_diagnose as s2d  # noqa: E402  (imports s2_headroom: g.RANGES = "strobl2021")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.rl import pqn  # noqa: E402

s2 = s2d.s2
s1, w4, g, wf = s2.s1, s2.w4, s2.g, s2.wf
CFG = pqn.PQNConfig(batch_patients=1024)  # arxiv-2407.04811 Table 4
N_TRAIN = 2048
SEEDS = s1.SEEDS_A
ARMS = {"mlp": (g.EP, None), "o_full": (s2.ep_for("o_full"), s2.make("o_full"))}
HEADROOM_D = s2.HEADROOM_D
SHARDS = g.OUT / "s3_shards"
POLICIES = g.OUT / "s3_policies"


def cell(arch: str, w: float, seed: int) -> dict:
    ep, make = ARMS[arch]
    pol, secs = w4.train_policy(arch, w, seed, 0.0, ep=ep, make=make, cfg=CFG, n_train=N_TRAIN)
    w4.save_policy(pol, POLICIES, arch, w, seed)
    tr = sample_cohort(g.FAM, N_TRAIN, seed=seed, ranges=g.RANGES)
    k = w4.eval_key(seed)
    train = s2d.score(pol, ep, jnp.asarray(tr.thetas), jnp.asarray(tr.y0s), k, w)
    return {
        "arch": arch,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        "train_cohort": train,
        **s1.tiers_eval(pol, seed, w, ep=ep),
    }


def spawn(arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def reading(ci, r: float) -> str:
    lo, hi = ci
    if lo > HEADROOM_D:
        return "headroom: the gate opens"
    if r < 0:
        return "still trainer-limited"
    if hi < 0:
        return "reward-limited"
    if hi < HEADROOM_D:
        return "no headroom: L1's mechanism gains support"
    return "inconclusive"


def renamed(paths, arch: str) -> list[dict]:
    return [{**json.loads(p.read_text()), "arch": arch} for p in paths]


def analyse(cells: list[dict]) -> dict:
    rules = [json.loads(p.read_text()) for p in sorted(s1.SHARDS.glob("ra_*.json"))]
    rules = [c for c in rules if c["seed"] in SEEDS]
    old = renamed(sorted(s1.SHARDS.glob("a_mlp_*.json")), "mlp32")
    old += renamed(sorted(s2.SHARDS.glob("a_o_full_*.json")), "o_full32")
    allc = cells + old
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in allc}

    def cmp(a, b, tier):
        r = w4.comparison(allc, valid, a, b, "level", SEEDS, (tier,))
        r["wilcoxon_greater_p"] = w4.wilcoxon_per_w_p(allc, valid, a, b, SEEDS, (tier,))
        return r

    by = {(c["arch"], c["w"], c["seed"]): c for c in cells}
    rd = np.array(
        [
            by[("o_full", w, s)]["train_cohort"]["return"]
            - by[("mlp", w, s)]["train_cohort"]["return"]
            for w in w4.WEIGHTS
            for s in SEEDS
        ]
    )
    res = {
        "prereg": "adaptive_therapy/PREREG.md S3 [2026-09-29]",
        "nonfinite": [str(k) for k, ok in valid.items() if not ok],
        "primary": cmp("o_full", "mlp", "tier0"),
        "tier1": cmp("o_full", "mlp", "tier1"),
        "R": {"mean": float(rd.mean()), "cells_positive": int((rd > 0).sum()), "n": len(rd)},
        "budget_effect": {
            "mlp": cmp("mlp", "mlp32", "tier0"),
            "o_full": cmp("o_full", "o_full32", "tier0"),
        },
        "table": w4.table(
            allc, valid, rules, ("mlp", "o_full", "mlp32", "o_full32"), SEEDS, s1.TIERS
        ),
    }
    res["reading"] = reading(res["primary"]["ci95"], res["R"]["mean"])
    return res


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    s2.reproduction_guard()
    jobs = [(a, w, s) for a in ARMS for w in w4.WEIGHTS for s in SEEDS]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(cells)
    for k in ("primary", "tier1"):
        print(f"O_full − MLP {k:8s} {s1.ci(res[k])}")
    for k, v in res["budget_effect"].items():
        print(f"budget effect {k:6s} (1024 − 32, Tier 0) {s1.ci(v)}")
    r = res["R"]
    print(
        f"R (train return, O_full − MLP) {r['mean']:+.2f}, higher in {r['cells_positive']}/{r['n']}"
    )
    print(f"READING: {res['reading']}")
    (g.OUT / "s3_budget.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        (SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(cell(arch, w, seed)))
        raise SystemExit(0)
    raise SystemExit(main())
