"""V10: state-dropout training of the LSTM and ODE-RNN (V9's treatment, recurrent arms).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-17] V10, before any cell
ran. In the episodes whose first observation is positive (about half) the
recurrent state is reset at every visit. Part A of S1: seeds 31-40, trained at
p = 0, Tiers 0/1, paired against S1's saved shards.

  *_show  memory always available at test    PRIMARY: lstm_sd - lstm, ode_sd - ode_rnn, Tier 1
  *_hide  memory never available at test

Run: JAX_PLATFORMS=cpu uv run python research/v10_statedropout.py
     -> research/outputs/v10_statedropout.json, v10_shards/, v10_policies/
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
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy  # noqa: E402
from adaptive_therapy.policies.state_dropout import StateDropout  # noqa: E402

w4, g, wf = s1.w4, s1.g, s1.wf
BASE = {"lstm_sd": "lstm", "ode_sd": "ode_rnn"}  # arm -> the S1 arm it is compared with
w4.ARCHS["lstm_sd"] = lambda k: StateDropout(LSTMPolicy(k, layer_norm=True))
w4.ARCHS["ode_sd"] = lambda k: StateDropout(ODERNNPolicy(k, layer_norm=True))
SHARDS = g.OUT / "v10_shards"
POLICIES = g.OUT / "v10_policies"
GATE = 10.0  # days; the program's gate, as registered
MODES = ("show", "hide")


def cell(arch: str, w: float, seed: int) -> dict:
    policy, secs = w4.train_policy(arch, w, seed, train_drop=0.0)
    w4.save_policy(policy, POLICIES, arch, w, seed)
    out = {"arch": arch, "w": w, "seed": seed, "train_seconds": secs}
    for mode in MODES:
        name = f"{arch}_{mode}"
        out[name] = {
            "arch": name,
            "w": w,
            "seed": seed,
            **s1.tiers_eval(policy.forced(mode), seed, w),
        }
    return out


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


def old_cells() -> list[dict]:
    a = [json.loads(f.read_text()) for f in sorted(s1.SHARDS.glob("a_*.json"))]
    return a


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
    names = [f"{c['arch']}_{m}" for c in new for m in MODES]
    cells = old_cells() + [c[f"{c['arch']}_{m}"] for c in new for m in MODES]
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells}
    res = {"prereg": "adaptive_therapy/PREREG.md V10 [2026-10-17]"}
    res["nonfinite"] = [
        f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok and a in set(names)
    ]
    res["vs"], res["reading"] = {}, {}
    for tier in s1.TIERS:
        for arm, base in BASE.items():
            for mode in MODES:
                a = f"{arm}_{mode}"
                res["vs"][f"{a}-{base}_{tier}"] = compare(cells, valid, a, base, tier)
                res["vs"][f"{a}-mlp_{tier}"] = compare(cells, valid, a, "mlp", tier)
            res["vs"][f"{arm}_hide-{arm}_show_{tier}"] = compare(
                cells, valid, f"{arm}_hide", f"{arm}_show", tier
            )
        res["vs"][f"ode_sd_show-lstm_sd_show_{tier}"] = compare(
            cells, valid, "ode_sd_show", "lstm_sd_show", tier
        )
    res["arms"] = {
        f"{a}_{t}": describe(cells, a, t)
        for a in ("mlp", "lstm", "ode_rnn", *(f"{x}_{m}" for x in BASE for m in MODES))
        for t in s1.TIERS
    }
    for arm, base in BASE.items():
        lo, hi = res["vs"][f"{arm}_show-{base}_tier1"]["ci95"]
        res["reading"][arm] = (
            "shortcut explains the deficit"
            if lo > GATE
            else "does not"
            if hi < GATE
            else "inconclusive"
        )
        lo, hi = res["vs"][f"{arm}_show-mlp_tier1"]["ci95"]
        res["reading"][f"{arm}_vs_mlp"] = "edge" if lo > 0 else "deficit" if hi < 0 else "ties"
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, sd) for a in ("ode_sd", "lstm_sd") for w in w4.WEIGHTS for sd in s1.SEEDS_A]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        new = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(new)
    (g.OUT / "v10_statedropout.json").write_text(json.dumps(res, indent=2))
    print(f"nonfinite {res['nonfinite']}")
    for k, v in res["vs"].items():
        print(f"  {k:30s} {ci(v)}")
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
