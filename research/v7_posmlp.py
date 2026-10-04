"""V7: the positional MLP (memoryless + clock) on S1's exact design.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-04] V7, before any cell ran.
Only the arm is new; seeds, cohorts, trainer, w grid, tiers and drops are S1's,
so every comparison pairs against S1's saved shards.

  A   seeds 31-40, trained at p = 0, Tiers 0/1.  Primary: Pos - MLP, Tier 1.
  B   seeds 41-50, trained at p = 0.35, W4b test. Primary: Pos - MLP, p in {.2, .5}.

Run: JAX_PLATFORMS=cpu uv run python research/v7_posmlp.py
     -> research/outputs/v7_posmlp.json, v7_shards/, v7_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets Strobl ranges; backend env first)
from v1_review import median_ci  # noqa: E402


from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402

w4, w4b, g, wf = s1.w4, s1.w4b, s1.g, s1.wf
ARCH = "pos_mlp"
w4.ARCHS[ARCH] = lambda k: MLPPolicy(k, width=386, layer_norm=True, clock=True)
SHARDS = g.OUT / "v7_shards"
POLICIES = g.OUT / "v7_policies"


def cell_a(w: float, seed: int) -> dict:
    policy, secs = w4.train_policy(ARCH, w, seed, train_drop=0.0)
    w4.save_policy(policy, POLICIES / "a", ARCH, w, seed)
    return {
        "arch": ARCH,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        **s1.tiers_eval(policy, seed, w),
    }


def cell_b(w: float, seed: int) -> dict:
    return w4.train_cell(ARCH, w, seed, POLICIES / "b")


WORKERS = {"a": cell_a, "b": cell_b}


def spawn(kind: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{kind}_{ARCH}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", kind, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {kind} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def s1_cells(kind: str) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(s1.SHARDS.glob(f"{kind}_*.json"))]


def compare(cells, valid, others, seeds, drops) -> dict:
    out = {}
    for x in others:
        d = w4.paired(cells, valid, ARCH, x, "level", seeds, drops)
        out[x] = {**w4.delta_ci(d, drops), "median": median_ci(d)}
    return out


def analyse(new_a, new_b) -> dict:
    a = s1_cells("a") + new_a
    valid_a = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in a}
    b = w4b.with_mtd(s1_cells("b") + new_b, s1_cells("rb"))
    valid_b = {(c["arch"], c["w"], c["seed"]): w4b.finite(c) for c in b}
    n = len(w4.WEIGHTS) * len(s1.SEEDS_A)
    bad = [k for k, ok in {**valid_a, **valid_b}.items() if k[0] == ARCH and not ok]
    res = {
        "prereg": "adaptive_therapy/PREREG.md V7 [2026-10-04]",
        "arm_valid": len(bad) / (2 * n) <= w4b.MAX_NONFINITE,
        "nonfinite": [f"{a}_w{w}_s{s}" for a, w, s in bad],
        "A": {
            t: compare(a, valid_a, ("mlp", "lstm", "ode_rnn"), s1.SEEDS_A, (t,)) for t in s1.TIERS
        },
        "B": compare(
            b, valid_b, ("mlp", "lstm", "lstm_dt", "ode_rnn", "MTD"), s1.SEEDS_B, w4.TEST_DROPS
        ),
        "B_p0.5_vs_MTD": compare(b, valid_b, ("MTD",), s1.SEEDS_B, (0.5,))["MTD"],
        "B_p0": compare(b, valid_b, ("mlp",), s1.SEEDS_B, (0.0,))["mlp"],
    }
    for key, r in (("A_tier1", res["A"]["tier1"]["mlp"]), ("B", res["B"]["mlp"])):
        lo, hi = r["ci95"]
        res[f"primary_{key}"] = "helps" if lo > 0 else "hurts" if hi < 0 else "ties"
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(k, w, s) for k, seeds in (("a", s1.SEEDS_A), ("b", s1.SEEDS_B))
            for w in w4.WEIGHTS for s in seeds]  # fmt: skip
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        out = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(
        [c for j, c in zip(jobs, out, strict=True) if j[0] == "a"],
        [c for j, c in zip(jobs, out, strict=True) if j[0] == "b"],
    )
    (g.OUT / "v7_posmlp.json").write_text(json.dumps(res, indent=2))
    print(f"arm valid {res['arm_valid']}  nonfinite {res['nonfinite']}")
    for t in s1.TIERS:
        for x, v in res["A"][t].items():
            print(f"  A {t}  Pos-{x:8s} {ci(v)}")
    for x, v in res["B"].items():
        print(f"  B p.2/.5 Pos-{x:8s} {ci(v)}")
    print(f"  B p=0.5 Pos-MTD {ci(res['B_p0.5_vs_MTD'])}   B p=0 Pos-MLP {ci(res['B_p0'])}")
    print(f"PRIMARY  A Tier 1: {res['primary_A_tier1']}   B Tier 4: {res['primary_B']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        kind, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        cell = WORKERS[kind](w, seed)
        (SHARDS / f"{kind}_{ARCH}_w{w}_s{seed}.json").write_text(json.dumps(cell))
    else:
        sys.exit(main())
