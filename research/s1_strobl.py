"""S1: W3c's and W4b's designs re-run on Strobl 2021's LV ranges.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-29] S1, before any S1
cell ran. Only the theta box changes (RESEARCH_LOG L703: 9.8% of provisional
patients lie inside Strobl's ranges); every other piece is imported.

  Part A    W3c's design: {MLP, LSTM, ODE-RNN} x w x seeds {31..40}, trained on
            regular visits, tested on Tier 0 and Tier 1. H_ct_S: ODE CI above
            0 against both MLP and LSTM on Tier 1.
  Part B    W4b's design exactly, seeds {41..50}. H_ct4b_S: ODE CI above 0
            against MLP, LSTM, LSTM+dt and MTD.
  Validity  intention-to-treat in both parts (w4b_itt.itt_validity).

A Part-A cell stores its tiers under the keys "tier0"/"tier1", so W4's paired
and table helpers, which read cell[str(p)], take the tiers as their `drops`.

Run: JAX_PLATFORMS=cpu uv run python research/s1_strobl.py
     -> research/outputs/s1_strobl.{json,png}, s1_shards/, s1_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w4b_itt as w4b  # noqa: E402  (imports phase2_gate: backend env first)

w4, g, wf = w4b.w4, w4b.g, w4b.wf
g.RANGES = "strobl2021"  # the one change S1 registers

import numpy as np  # noqa: E402

from adaptive_therapy.policies.rules import RULES, RulePolicy  # noqa: E402

SEEDS_A = tuple(range(31, 41))
SEEDS_B = tuple(range(41, 51))
ARCHS_A = ("mlp", "lstm", "ode_rnn")
TIERS = ("tier0", "tier1")
SHARDS = g.OUT / "s1_shards"
POLICIES = g.OUT / "s1_policies"


def tiers_eval(policy, seed: int, w: float, ep=None) -> dict:
    out = {}
    for tier, cohort in (("tier0", w4.tier0_cohort), ("tier1", g.tier1_cohort)):
        ev = w4.evaluate(policy, seed, w, cohort=cohort, drops=(0.0,), ep=ep)
        out[tier] = ev["0.0"]
        out[f"closed_loop_{tier}"] = ev["closed_loop"]
    out["closed_loop"] = out["closed_loop_tier1"]
    return out


def cell_a(arch: str, w: float, seed: int) -> dict:
    policy, secs = w4.train_policy(arch, w, seed, train_drop=0.0)
    w4.save_policy(policy, POLICIES / "a", arch, w, seed)
    return {
        "arch": arch,
        "w": w,
        "seed": seed,
        "train_seconds": secs,
        **tiers_eval(policy, seed, w),
    }


def cell_b(arch: str, w: float, seed: int) -> dict:
    return w4.train_cell(arch, w, seed, POLICIES / "b")


def rule_a(kind: str, seed: int) -> dict:
    return {"arch": kind, "seed": seed, **tiers_eval(RulePolicy(kind), seed, 0.0)}


WORKERS = {
    "a": lambda arch, w, s: cell_a(arch, w, s),
    "b": lambda arch, w, s: cell_b(arch, w, s),
    "ra": lambda arch, w, s: rule_a(arch, s),
    "rb": lambda arch, w, s: w4.rules_cell(arch, s),
}


def spawn(kind: str, arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", kind, arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {kind:2s} {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def finite_a(cell: dict) -> bool:
    return all(np.isfinite(cell[t][k]).all() for t in TIERS for k in ("ttp_capped", "exposure"))


def analyse_a(cells, rules) -> dict:
    valid = {(c["arch"], c["w"], c["seed"]): finite_a(c) for c in cells}
    n = len(w4.WEIGHTS) * len(SEEDS_A)
    arm_valid = {
        a: sum(not valid[(a, w, s)] for w in w4.WEIGHTS for s in SEEDS_A) / n <= w4b.MAX_NONFINITE
        for a in ARCHS_A
    }

    def cmp(a, b, tier):
        r = w4.comparison(cells, valid, a, b, "level", SEEDS_A, (tier,))
        r["wilcoxon_greater_p"] = w4.wilcoxon_per_w_p(cells, valid, a, b, SEEDS_A, (tier,))
        return r

    res = {
        "arm_valid": arm_valid,
        "nonfinite": [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok],
        "table": w4.table(cells, valid, rules, ARCHS_A, SEEDS_A, TIERS),
        "closed_loop": {a: [c["closed_loop"] for c in cells if c["arch"] == a] for a in ARCHS_A},
    }
    r = {"valid": all(arm_valid.values()), "vs": {}}
    for x in ("mlp", "lstm"):
        r["vs"][x] = {t: cmp("ode_rnn", x, t) for t in TIERS}
    r["passes"] = bool(r["valid"] and min(v["tier1"]["ci95"][0] for v in r["vs"].values()) > 0)
    res["H_ct_S"] = r
    res["memory"] = {t: cmp("lstm", "mlp", t) for t in TIERS}
    return res


def ci(v: dict) -> str:
    return f"{v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"


def report_a(res: dict) -> None:
    r = res["H_ct_S"]
    for t in TIERS:
        print(
            f"  {t}: "
            + "  ".join(f"ODE−{x} {ci(v[t])}" for x, v in r["vs"].items())
            + f"  LSTM−MLP {ci(res['memory'][t])}"
        )
    print(f"H_ct_S (Tier 1) valid={r['valid']} PASS={r['passes']}  nonfinite {res['nonfinite']}")
    for a, cl in res["closed_loop"].items():
        print(
            f"  closed-loop {a:8s} median {np.median(cl):.3f}  <0.10: {np.mean(np.array(cl) < 0.10):.0%}"
        )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = (
        [("a", a, w, s) for a in ARCHS_A for w in w4.WEIGHTS for s in SEEDS_A]
        + [("ra", r, 0.0, s) for r in RULES for s in SEEDS_A]
        + [("b", a, w, s) for a in w4.ARCHS for w in w4.WEIGHTS for s in SEEDS_B]
        + [("rb", r, 0.0, s) for r in RULES for s in SEEDS_B]
    )
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        out = list(ex.map(lambda j: spawn(*j), jobs))
    part = {k: [c for j, c in zip(jobs, out, strict=True) if j[0] == k] for k in WORKERS}
    res = {"prereg": "adaptive_therapy/PREREG.md S1 [2026-09-29]", "ranges": g.RANGES}
    res["A"] = analyse_a(part["a"], part["ra"])
    print("== S1 Part A: Tiers 0/1, regular visits =="), report_a(res["A"])
    res["B"] = w4b.analyse(part["b"], part["rb"], SEEDS_B)
    print("== S1 Part B: Tier 4 (W4b design) =="), w4b.report(res["B"])
    (g.OUT / "s1_strobl.json").write_text(json.dumps(res, indent=2))
    rows = (("B", "S1 Part B, Strobl ranges"), ("A4", "W4b, provisional ranges"))
    prov = json.loads((g.OUT / "w4b_itt.json").read_text())["table"]
    w4.plot({"B_table": res["B"]["table"], "A4_table": prov}, g.OUT / "s1_strobl.png", rows, "S1")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        kind, arch, w, seed = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        cell = WORKERS[kind](arch, w, seed)
        (SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json").write_text(json.dumps(cell))
        raise SystemExit(0)
    raise SystemExit(main())
