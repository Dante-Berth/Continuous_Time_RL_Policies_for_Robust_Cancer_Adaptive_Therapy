"""V11: LSTM capacity sweep (is the recurrent deficit an under-powered network?).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-17] V11, before any cell
ran. S1 Part A's design with the LSTM at 64 and 128 units (S1 had 26): seeds
31-40, trained at p = 0, Tiers 0/1, w in {0.05, 0.1, 0.2, 0.4}; paired against
S1's saved shards (same seeds, same cohorts).

Run: JAX_PLATFORMS=cpu uv run python research/v11_capacity.py
     -> research/outputs/v11_capacity.json, v11_shards/, v11_policies/
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

from adaptive_therapy.policies.lstm import LSTMPolicy  # noqa: E402

w4, g, wf = s1.w4, s1.g, s1.wf
UNITS = {"lstm_64": 64, "lstm_128": 128}
for _name, _h in UNITS.items():
    w4.ARCHS[_name] = lambda k, h=_h: LSTMPolicy(k, hidden=h, layer_norm=True)
SHARDS = g.OUT / "v11_shards"
POLICIES = g.OUT / "v11_policies"
GATE = 10.0  # days; the program's gate, as registered


def cell(arch: str, w: float, seed: int) -> dict:
    policy, secs = w4.train_policy(arch, w, seed, train_drop=0.0)
    w4.save_policy(policy, POLICIES, arch, w, seed)
    return {"arch": arch, "w": w, "seed": seed, "train_seconds": secs, **s1.tiers_eval(policy, seed, w)}


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
    }


def analyse(new: list[dict]) -> dict:
    old = [json.loads(f.read_text()) for f in sorted(s1.SHARDS.glob("a_*.json"))]
    cells = old + new
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells}
    res = {"prereg": "adaptive_therapy/PREREG.md V11 [2026-10-17]", "vs": {}, "reading": {}}
    res["nonfinite"] = [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok and a in UNITS]
    for tier in s1.TIERS:
        for a in UNITS:
            res["vs"][f"{a}-lstm_{tier}"] = compare(cells, valid, a, "lstm", tier)
            res["vs"][f"{a}-mlp_{tier}"] = compare(cells, valid, a, "mlp", tier)
        res["vs"][f"lstm_128-lstm_64_{tier}"] = compare(cells, valid, "lstm_128", "lstm_64", tier)
    res["arms"] = {
        f"{a}_{t}": describe(cells, a, t) for a in ("mlp", "lstm", *UNITS) for t in s1.TIERS
    }
    lo = max(res["vs"][f"{a}-lstm_tier1"]["ci95"][0] for a in UNITS)
    hi = max(res["vs"][f"{a}-lstm_tier1"]["ci95"][1] for a in UNITS)
    res["reading"]["capacity"] = (
        "capacity-limited" if lo > GATE else "not capacity-limited" if hi < GATE else "inconclusive"
    )
    for a in UNITS:
        lo_, hi_ = res["vs"][f"{a}-mlp_tier1"]["ci95"]
        res["reading"][f"{a}_vs_mlp"] = "edge" if lo_ > 0 else "deficit" if hi_ < 0 else "ties"
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, sd) for a in UNITS for w in w4.WEIGHTS for sd in s1.SEEDS_A]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        new = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(new)
    (g.OUT / "v11_capacity.json").write_text(json.dumps(res, indent=2))
    print(f"nonfinite {res['nonfinite']}")
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
