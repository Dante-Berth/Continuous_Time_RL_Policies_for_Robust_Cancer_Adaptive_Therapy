"""S2: full-state headroom bound -- how much does seeing the hidden state gain?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-29] S2, before any S2
cell ran. Owner-approved exception to burden-only: the oracles are REFERENCES,
never arms, and are never reported as policies.

  O_state   S1's MLP-441 fed [noisy log burden, log S/b0, log R/b0]
  O_full    ... plus theta (rollout.oracle_features); its information contains
            that of any burden-only policy, recurrent or not.
  Pairing   S1's seeds, cohorts, eval keys, noise and visit masks (Strobl
            ranges); the comparators are S1's EXISTING cells, re-read from
            s1_shards/. The script first re-scores two saved S1 MLP policies and
            refuses to run if either differs from its shard: the oracle hook
            must leave burden-only rollouts bit-identical.
  Primary   Part A, Tier 0: Delta(O_full, MLP), seed+patient CI. CI < 0 =>
            trainer cannot use the input; hi < +10 d => no headroom; lo > +10 d
            => headroom; else inconclusive.
  Part B    W4b design: Delta(O_full, MTD) at p = 0.5; lo > 0 => full
            information restores feedback's value, hi < 0 => the collapse is
            inherent to missed visits under hold.

Run: JAX_PLATFORMS=cpu uv run python research/s2_headroom.py
     -> research/outputs/s2_headroom.{json,log}, s2_shards/, s2_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets g.RANGES = "strobl2021")

import jax  # noqa: E402

from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402
from adaptive_therapy.rl.rollout import ORACLE_DIMS  # noqa: E402

w4, w4b, g, wf = s1.w4, s1.w4b, s1.g, s1.wf
ORACLES = {"o_state": "state", "o_full": "full"}
HEADROOM_D = 10.0  # days; a quarter of the MLP's Tier-0 margin over MTD in S1
SHARDS = g.OUT / "s2_shards"
POLICIES = g.OUT / "s2_policies"


def ep_for(arch: str):
    return replace(g.EP, oracle=ORACLES[arch])


def make(arch: str):
    dim = 1 + ORACLE_DIMS[ORACLES[arch]]
    return lambda k: MLPPolicy(k, width=441, layer_norm=True, in_dim=dim)


def cell_a(arch: str, w: float, seed: int) -> dict:
    pol, secs = w4.train_policy(arch, w, seed, 0.0, ep=ep_for(arch), make=make(arch))
    w4.save_policy(pol, POLICIES / "a", arch, w, seed)
    return {
        "arch": arch,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        **s1.tiers_eval(pol, seed, w, ep=ep_for(arch)),
    }


def cell_b(arch: str, w: float, seed: int) -> dict:
    pol, secs = w4.train_policy(arch, w, seed, w4.TRAIN_DROP, ep=ep_for(arch), make=make(arch))
    w4.save_policy(pol, POLICIES / "b", arch, w, seed)
    return {
        "arch": arch,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        **w4.evaluate(pol, seed, w, ep=ep_for(arch)),
    }


WORKERS = {"a": cell_a, "b": cell_b}


def spawn(kind: str, arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", kind, arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {kind} {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def s1_shard(name: str) -> dict:
    return json.loads((s1.SHARDS / f"{name}.json").read_text())


def reproduction_guard() -> None:
    """S1's saved MLP policies must score exactly as S1 recorded them."""
    import equinox as eqx

    like = w4.ARCHS["mlp"](jax.random.PRNGKey(0))
    for part, w, seed in (("a", 0.05, 31), ("b", 0.05, 41)):
        pol = eqx.tree_deserialise_leaves(s1.POLICIES / part / f"mlp_w{w}_s{seed}.eqx", like)
        ref = s1_shard(f"{part}_mlp_w{w}_s{seed}")
        if part == "a":
            now, ref_d = s1.tiers_eval(pol, seed, w)["tier0"], ref["tier0"]
        else:
            now, ref_d = w4.evaluate(pol, seed, w)["0.5"], ref["0.5"]
        for k in ("ttp_capped", "exposure"):
            if now[k] != ref_d[k]:
                raise RuntimeError(f"S1 {part} mlp w{w} s{seed}: {k} moved; burden-only changed")
    print("reproduction guard: S1 MLP cells re-score bit-identically", flush=True)


def reading_primary(ci) -> str:
    lo, hi = ci
    if hi < 0:
        return "trainer cannot use the input (no headroom claim)"
    if hi < HEADROOM_D:
        return "no headroom"
    if lo > HEADROOM_D:
        return "headroom"
    return "inconclusive"


def reading_b(ci) -> str:
    lo, hi = ci
    if lo > 0:
        return "full information restores feedback's value at 50% missed"
    if hi < 0:
        return "collapse inherent to missed visits under hold"
    return "inconclusive"


def analyse(oracle_a, oracle_b) -> dict:
    s1_a = [s1_shard(p.stem) for p in sorted(s1.SHARDS.glob("a_mlp_*.json"))]
    rules_a = [s1_shard(p.stem) for p in sorted(s1.SHARDS.glob("ra_*.json"))]
    s1_b = [s1_shard(p.stem) for p in sorted(s1.SHARDS.glob("b_*.json"))]
    rules_b = [s1_shard(p.stem) for p in sorted(s1.SHARDS.glob("rb_*.json"))]

    cells_a = s1_a + oracle_a
    valid_a = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells_a}
    cells_b = w4b.with_mtd(s1_b + oracle_b, rules_b)
    valid_b = {(c["arch"], c["w"], c["seed"]): w4b.finite(c) for c in cells_b}

    def cmp_a(a, b, tier):
        r = w4.comparison(cells_a, valid_a, a, b, "level", s1.SEEDS_A, (tier,))
        r["wilcoxon_greater_p"] = w4.wilcoxon_per_w_p(cells_a, valid_a, a, b, s1.SEEDS_A, (tier,))
        return r

    def cmp_b(a, b, drops):
        return w4.comparison(cells_b, valid_b, a, b, "level", s1.SEEDS_B, drops)

    n = len(w4.WEIGHTS) * (len(s1.SEEDS_A) + len(s1.SEEDS_B))
    bad = [k for v in (valid_a, valid_b) for k, ok in v.items() if not ok]
    res = {
        "prereg": "adaptive_therapy/PREREG.md S2 [2026-09-29]",
        "nonfinite": [f"{k}" for k in bad],
        "arm_valid": {o: sum(k[0] == o for k in bad) / n <= w4b.MAX_NONFINITE for o in ORACLES},
        "A": {f"{o}-mlp|{t}": cmp_a(o, "mlp", t) for o in ORACLES for t in s1.TIERS},
        "B": {
            "o_full-MTD|0.5": cmp_b("o_full", "MTD", (0.5,)),
            "o_full-lstm|0.5": cmp_b("o_full", "lstm", (0.5,)),
            "o_full-mlp|test": cmp_b("o_full", "mlp", w4.TEST_DROPS),
            "o_state-MTD|0.5": cmp_b("o_state", "MTD", (0.5,)),
        },
        "table_a": w4.table(cells_a, valid_a, rules_a, ("mlp", *ORACLES), s1.SEEDS_A, s1.TIERS),
        "table_b": w4.table(cells_b, valid_b, rules_b, ("mlp", "lstm", *ORACLES), s1.SEEDS_B),
        "closed_loop": {o: [c["closed_loop"] for c in oracle_a if c["arch"] == o] for o in ORACLES},
    }
    res["primary_reading"] = reading_primary(res["A"]["o_full-mlp|tier0"]["ci95"])
    res["B_reading"] = reading_b(res["B"]["o_full-MTD|0.5"]["ci95"])
    return res


def report(res: dict) -> None:
    for k, v in (*res["A"].items(), *res["B"].items()):
        print(f"  {k:18s} {s1.ci(v)}")
    print(f"PRIMARY (O_full − MLP, Tier 0): {res['primary_reading']}")
    print(f"PART B (O_full − MTD, p 0.5): {res['B_reading']}")
    print(f"arm valid {res['arm_valid']}  nonfinite {res['nonfinite']}")


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    reproduction_guard()
    jobs = [
        (part, o, w, s)
        for part, seeds in (("a", s1.SEEDS_A), ("b", s1.SEEDS_B))
        for o in ORACLES
        for w in w4.WEIGHTS
        for s in seeds
    ]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        out = list(ex.map(lambda j: spawn(*j), jobs))
    split = {k: [c for j, c in zip(jobs, out, strict=True) if j[0] == k] for k in WORKERS}
    res = analyse(split["a"], split["b"])
    print("== S2: full-state headroom bound =="), report(res)
    (g.OUT / "s2_headroom.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        kind, arch, w, seed = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        cell = WORKERS[kind](arch, w, seed)
        (SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json").write_text(json.dumps(cell))
        raise SystemExit(0)
    raise SystemExit(main())
