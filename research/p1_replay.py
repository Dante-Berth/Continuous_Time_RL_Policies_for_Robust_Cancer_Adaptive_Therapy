"""P1: do the paper's two negatives hold on the 55 Bruchovsky-fitted patients?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-01] P1, before any replay
ran. No retraining: S1's saved policies (Parts A and B) and the rules, scored on
R2's fits (r^2 >= 0.5) with S1's episode, U, ITT rule and seed+patient CI. Only
the test cohort changes; it is the same 55 fits for every seed.

  H1  Part A, p 0: ODE-MLP and ODE-LSTM; no CI lower bound > 0 => no edge replicates.
  H2  Part B, p 0.5: each learned arm - MTD; no CI lower bound > 0 => replicated.

Run: JAX_PLATFORMS=cpu uv run python research/p1_replay.py
     -> research/outputs/p1_replay.{json,log}
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets g.RANGES = "strobl2021"; backend env first)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.lotka_volterra import initial_state  # noqa: E402
from adaptive_therapy.envs.sampling import STROBL_D_D, STROBL_R_S  # noqa: E402
from adaptive_therapy.policies.rules import RulePolicy  # noqa: E402

w4, w4b, g, wf = s1.w4, s1.w4b, s1.g, s1.wf
FITS = g.OUT / "r2_bruchovsky_fit.json"
RESULT = g.OUT / "p1_replay.json"
SHARDS = g.OUT / "p1_shards"
K = 1e4  # scale only: the observation is burden / baseline
PARTS = {"a": (s1.ARCHS_A, s1.SEEDS_A), "b": (tuple(w4.ARCHS), s1.SEEDS_B)}
RULES = ("none", "MTD", "AT50", "thr0.8")
PSEUDO = ("MTD", "thr0.8")  # rules compared against arms, formed per w


def twins() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thetas, y0s, in_box) for R2's kept fits, in R2's patient order."""
    kept = [f for f in json.loads(FITS.read_text())["fits"] if f["r2"] >= 0.5]
    th = np.array(
        [
            [
                STROBL_R_S,
                f["r_R"] * STROBL_R_S,
                K,
                f["d_T"] * STROBL_R_S,
                f["d_T"] * STROBL_R_S,
                STROBL_D_D,
            ]
            for f in kept
        ]
    )
    y0 = np.stack(
        [np.asarray(initial_state(t, f["f_R"], f["n0"])) for t, f in zip(th, kept, strict=True)]
    )
    box = np.array([f["r_R"] >= 0.5 and f["d_T"] <= 0.5 for f in kept])
    return th, y0, box


TH, Y0, IN_BOX = twins()
g.N_TEST = len(TH)  # every helper below reads the cohort size from here


def cohort(n: int, seed: int):
    return TH, Y0


def score(policy, seed: int, w: float) -> dict:
    return w4.evaluate(policy, seed, w, cohort=cohort, drops=w4.DROPS)


def worker(part: str, arch: str) -> list[dict]:
    """Every saved policy of one (part, arch), or the rules ("rules"), on the fits."""
    _, seeds = PARTS[part]
    if arch == "rules":
        return [
            {"arch": k, "seed": s, **score(RulePolicy(k), s, 0.0)} for k in RULES for s in seeds
        ]
    like = w4.ARCHS[arch](jax.random.PRNGKey(0))
    out = []
    for w in w4.WEIGHTS:
        for s in seeds:
            pol = eqx.tree_deserialise_leaves(s1.POLICIES / part / f"{arch}_w{w}_s{s}.eqx", like)
            out.append({"arch": arch, "w": w, "seed": s, **score(pol, s, w)})
    return out


def spawn(part: str, arch: str) -> list[dict]:
    """One process per (part, arch). 2026-10-01: one process scoring all 280
    policies recompiled every rollout and died of LLVM "Cannot allocate memory"
    after two arms -- the vm.max_map_count leak of RESEARCH_LOG L207."""
    shard = SHARDS / f"{part}_{arch}.json"
    if not shard.exists():  # resume: a finished shard is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", part, arch],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done part {part}: {arch}", flush=True)
    return json.loads(shard.read_text())


def cells_for(part: str) -> tuple[list[dict], list[dict]]:
    archs, _ = PARTS[part]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        shards = list(ex.map(lambda a: spawn(part, a), (*archs, "rules")))
    return [c for sh in shards[:-1] for c in sh], shards[-1]


def subset(cells: list[dict], idx: np.ndarray) -> list[dict]:
    """The same cells restricted to patients `idx` (for the in-box re-analysis)."""
    keys = ("ttp_capped", "censored", "exposure", "burden_integral")
    return [
        {**c, **{str(p): {k: list(np.asarray(c[str(p)][k])[idx]) for k in keys} for p in w4.DROPS}}
        for c in cells
    ]


def analyse(cells, rules, part: str) -> dict:
    archs, seeds = PARTS[part]
    allc = cells + [{**r, "w": w} for r in rules if r["arch"] in PSEUDO for w in w4.WEIGHTS]
    valid, arm_valid = w4b.itt_validity(allc, (*archs, *PSEUDO), seeds)

    def cmp(a, b, p):
        r = w4.comparison(allc, valid, a, b, "level", seeds, (p,))
        r["wilcoxon_greater_p"] = w4.wilcoxon_per_w_p(allc, valid, a, b, seeds, (p,))
        return r

    res = {
        "arm_valid": arm_valid,
        "nonfinite": [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok],
        "table": w4.table(cells, valid, rules, archs, seeds),
    }
    if part == "a":
        res["H1"] = {x: cmp("ode_rnn", x, 0.0) for x in ("mlp", "lstm")}
        res["mlp_minus_thr08"] = cmp("mlp", "thr0.8", 0.0)
    else:
        res["H2"] = {a: cmp(a, "MTD", 0.5) for a in archs}
    return res


def reading(res: dict) -> dict:
    lo1 = [v["ci95"][0] for v in res["a"]["H1"].values()]
    h1 = (
        "no continuous-time edge replicates on fitted patients"
        if max(lo1) <= 0
        else "an edge appears on fitted patients"
        if min(lo1) > 0
        else "mixed"
    )
    beat = [a for a, v in res["b"]["H2"].items() if v["ci95"][0] > 0]
    h2 = "no learned arm beats MTD at 50% missed, replicated" if not beat else f"beats MTD: {beat}"
    return {"H1": h1, "H2": h2}


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    res = {"prereg": "adaptive_therapy/PREREG.md P1 [2026-10-01]", "n_patients": int(g.N_TEST)}
    raw = {part: cells_for(part) for part in PARTS}
    res.update({part: analyse(*raw[part], part) for part in PARTS})
    res["reading"] = reading(res)
    idx = np.flatnonzero(IN_BOX)
    g.N_TEST = len(idx)
    res["in_box"] = {
        "n_patients": len(idx),
        **{part: analyse(subset(c, idx), subset(r, idx), part) for part, (c, r) in raw.items()},
    }
    for k in ("table",):
        for part in PARTS:
            res["in_box"][part].pop(k)
    res["in_box"]["reading"] = reading(res["in_box"])
    for tag, r in (("all 55", res), (f"in box ({len(idx)})", res["in_box"])):
        print(f"\n== {tag}")
        for x, v in r["a"]["H1"].items():
            print(f"H1 ODE − {x:5s} p0   {s1.ci(v)}")
        for a, v in r["b"]["H2"].items():
            print(f"H2 {a:8s} − MTD p0.5 {s1.ci(v)}")
        print(f"MLP − thr0.8 p0     {s1.ci(r['a']['mlp_minus_thr08'])}")
        print(f"READING: {r['reading']}")
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        part, arch = sys.argv[2], sys.argv[3]
        (SHARDS / f"{part}_{arch}.json").write_text(json.dumps(worker(part, arch)))
        raise SystemExit(0)
    raise SystemExit(main())
