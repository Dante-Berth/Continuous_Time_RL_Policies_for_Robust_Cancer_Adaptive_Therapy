"""V9: clock-dropout training of the positional MLP (V7's arm, one change).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-17] V9, before any cell
ran. Tests whether V7's deficit is the trainer fitting a clock-keyed schedule:
`cd_mlp` has its clock hidden in the episodes whose first observation is
positive (about half), so it must learn feedback. Part A of S1/V7: seeds 31-40,
trained at p = 0, Tiers 0/1, paired against S1's and V7's saved shards.

  cd_show  clock always shown at test    PRIMARY: cd_show - pos_mlp, Tier 1
  cd_hide  clock never shown at test

Run: JAX_PLATFORMS=cpu uv run python research/v9_clockdropout.py
     -> research/outputs/v9_clockdropout.json, v9_shards/, v9_policies/
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

import numpy as np  # noqa: E402

from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402

w4, g, wf = s1.w4, s1.g, s1.wf
ARCH = "cd_mlp"
w4.ARCHS[ARCH] = lambda k: MLPPolicy(k, width=386, layer_norm=True, clock_dropout=True)
SHARDS = g.OUT / "v9_shards"
POLICIES = g.OUT / "v9_policies"
GATE = 10.0  # days; the program's gate, as registered
MODES = {"cd_show": "show", "cd_hide": "hide"}
OTHERS = ("pos_mlp", "mlp", "lstm", "ode_rnn")


def cell(w: float, seed: int) -> dict:
    policy, secs = w4.train_policy(ARCH, w, seed, train_drop=0.0)
    w4.save_policy(policy, POLICIES, ARCH, w, seed)
    out = {"w": w, "seed": seed, "train_seconds": secs}
    for name, mode in MODES.items():
        out[name] = {"arch": name, "w": w, "seed": seed, **s1.tiers_eval(policy.forced(mode), seed, w)}
    return out


def spawn(w: float, seed: int) -> dict:
    shard = SHARDS / f"w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def old_cells() -> list[dict]:
    a = [json.loads(f.read_text()) for f in sorted(s1.SHARDS.glob("a_*.json"))]
    v7 = [json.loads(f.read_text()) for f in sorted((g.OUT / "v7_shards").glob("a_pos_mlp_*.json"))]
    return a + v7


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
    }


def analyse(new: list[dict]) -> dict:
    cells = old_cells() + [c[m] for c in new for m in MODES]
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells}
    res = {"prereg": "adaptive_therapy/PREREG.md V9 [2026-10-17]"}
    res["nonfinite"] = [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok and a in MODES]
    res["vs"] = {}
    for tier in s1.TIERS:
        for a in MODES:
            for b in OTHERS:
                res["vs"][f"{a}-{b}_{tier}"] = compare(cells, valid, a, b, tier)
        res["vs"][f"cd_hide-cd_show_{tier}"] = compare(cells, valid, "cd_hide", "cd_show", tier)
        res["vs"][f"pos_mlp-mlp_{tier}"] = compare(cells, valid, "pos_mlp", "mlp", tier)
    res["arms"] = {
        f"{a}_{t}": describe(cells, a, t) for a in (*MODES, *OTHERS) for t in s1.TIERS
    }
    lo, hi = res["vs"]["cd_show-pos_mlp_tier1"]["ci95"]
    res["primary_cd_show_vs_pos_tier1"] = (
        "shortcut confirmed" if lo > GATE else "refuted" if hi < GATE else "inconclusive"
    )
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(w, s) for w in w4.WEIGHTS for s in s1.SEEDS_A]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        new = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(new)
    (g.OUT / "v9_clockdropout.json").write_text(json.dumps(res, indent=2))
    print(f"nonfinite {res['nonfinite']}")
    for k, v in res["vs"].items():
        print(f"  {k:28s} {ci(v)}")
    for k, v in res["arms"].items():
        print(f"  {k:16s} {v}")
    print(f"PRIMARY cd_show - pos_mlp, Tier 1: {res['primary_cd_show_vs_pos_tier1']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        w, seed = float(sys.argv[2]), int(sys.argv[3])
        (SHARDS / f"w{w}_s{seed}.json").write_text(json.dumps(cell(w, seed)))
    else:
        sys.exit(main())
