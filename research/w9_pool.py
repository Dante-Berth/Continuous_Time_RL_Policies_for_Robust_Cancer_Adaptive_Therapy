"""W9: scheduled visits, take 2 -- train on a POOL of schedules.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-17] W9, before any cell
ran. W8's design (research/w8_scheduled.py) with one change: a schedule is drawn
per training patient from a pool, so the lookahead cannot be a fixed phase.

  Train   pool of 5 schedules (gaps <= 3).   Test (no retraining)  A = (1,3) in
          the pool; held out C = (1,2,3), D = (4,), E = (1,1,4).
  Check   mlp_L - mlp_blind over C, D, E: GATE, usable iff lower CI > 0.
  P1      ode_L - mlp_L over C, D, E.

Run: JAX_PLATFORMS=cpu uv run python research/w9_pool.py
     -> research/outputs/w9_pool.json, w9_shards/, w9_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets Strobl ranges; backend env first)
from v1_review import median_ci  # noqa: E402

import numpy as np  # noqa: E402

from adaptive_therapy.policies.lstm import LSTMPolicy  # noqa: E402
from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy  # noqa: E402
from adaptive_therapy.policies.rules import RulePolicy  # noqa: E402

w4, w4b, g, wf = s1.w4, s1.w4b, s1.g, s1.wf
SEEDS = tuple(range(71, 81))
POOL = ((1, 3), (2, 2), (1, 1, 2), (3, 1), (1, 2))
SCHEDULES = {"sA": (1, 3), "sC": (1, 2, 3), "sD": (4,), "sE": (1, 1, 4)}
HELD_OUT = ("sC", "sD", "sE")
LONG_GAP = ("sD", "sE")
SHARDS = g.OUT / "w9_shards"
POLICIES = g.OUT / "w9_policies"
# arm -> (constructor, sees lookahead)
ARMS = {
    "mlp_L": (lambda k: MLPPolicy(k, width=386, layer_norm=True, in_dim=2), True),
    "lstm_L": (lambda k: LSTMPolicy(k, hidden=25, layer_norm=True, dt_input=True, in_dim=2), True),
    "ode_L": (lambda k: ODERNNPolicy(k, width=62, layer_norm=True, in_dim=2), True),
    "mlp_blind": (lambda k: MLPPolicy(k, width=441, layer_norm=True), False),
}
RULES = ("MTD", "thr0.8")


def ep_for(sched: str, look: bool):
    return replace(g.EP, schedule=SCHEDULES[sched], lookahead=look)


def ep_train(look: bool):
    return replace(g.EP, schedule_pool=POOL, lookahead=look)


def eval_schedules(policy, seed: int, w: float, look: bool) -> dict:
    out = {}
    for s in SCHEDULES:
        ev = w4.evaluate(policy, seed, w, drops=(0.0,), ep=ep_for(s, look))
        out[s] = ev["0.0"]
        out[f"closed_loop_{s}"] = ev["closed_loop"]
    return out


def arm_cell(arch: str, w: float, seed: int) -> dict:
    make, look = ARMS[arch]
    policy, secs = w4.train_policy(
        arch, w, seed, train_drop=0.0, ep=ep_train(look), make=make
    )
    w4.save_policy(policy, POLICIES, arch, w, seed)
    return {"arch": arch, "w": w, "seed": seed, "train_seconds": secs,
            **eval_schedules(policy, seed, w, look)}  # fmt: skip


def rule_cell(kind: str, w: float, seed: int) -> dict:
    return {"arch": kind, "w": w, "seed": seed, **eval_schedules(RulePolicy(kind), seed, w, False)}


def worker(arch: str, w: float, seed: int) -> dict:
    return (rule_cell if arch in RULES else arm_cell)(arch, w, seed)


def spawn(arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {arch:9s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def finite(c: dict) -> bool:
    return all(np.isfinite(c[s][k]).all() for s in SCHEDULES for k in ("ttp_capped", "exposure"))


def describe(cells, arch, sched) -> dict:
    cs = [c for c in cells if c["arch"] == arch]
    pool = lambda k: np.concatenate([np.asarray(c[sched][k]) for c in cs])  # noqa: E731
    q = lambda x: [float(v) for v in np.percentile(x, [25, 50, 75])]  # noqa: E731
    cl = [c[f"closed_loop_{sched}"] for c in cs]
    return {
        "ttp_capped_q25_50_75": q(pool("ttp_capped")),
        "exposure_q25_50_75": q(pool("exposure")),
        "burden_integral_q25_50_75": q(pool("burden_integral")),
        "closed_loop_mean": float(np.mean(cl)),
    }


def compare(cells, valid, a, b, drops) -> dict:
    d = w4.paired(cells, valid, a, b, "level", SEEDS, drops)
    return {
        **w4.delta_ci(d, drops),
        "median": median_ci(d),
        "wilcoxon_per_w_p": w4.wilcoxon_per_w_p(cells, valid, a, b, SEEDS, drops),
    }


def reading(r: dict) -> str:
    lo, hi = r["ci95"]
    return "edge" if lo > 0 else "deficit" if hi < 0 else "ties"


def analyse(cells: list[dict]) -> dict:
    valid = {(c["arch"], c["w"], c["seed"]): finite(c) for c in cells}
    n = len(w4.WEIGHTS) * len(SEEDS)
    arm_valid = {
        a: sum(not valid[(a, w, s)] for w in w4.WEIGHTS for s in SEEDS) / n <= w4b.MAX_NONFINITE
        for a in ARMS
    }
    res = {
        "prereg": "adaptive_therapy/PREREG.md W9 [2026-10-17]",
        "arm_valid": arm_valid,
        "nonfinite": [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok],
        "manipulation_check": compare(cells, valid, "mlp_L", "mlp_blind", HELD_OUT),
        "P1_held_out": compare(cells, valid, "ode_L", "mlp_L", HELD_OUT),
        "P1_long_gaps": compare(cells, valid, "ode_L", "mlp_L", LONG_GAP),
        "P_trained": compare(cells, valid, "ode_L", "mlp_L", ("sA",)),
        "ode_vs_lstm": {s: compare(cells, valid, "ode_L", "lstm_L", (s,)) for s in SCHEDULES},
        "ode_vs_lstm_held_out": compare(cells, valid, "ode_L", "lstm_L", HELD_OUT),
        "vs_rules": {
            f"{a}-{r}_{s}": compare(cells, valid, a, r, (s,))
            for a in ARMS
            for r in RULES
            for s in SCHEDULES
        },
        "arms": {f"{a}_{s}": describe(cells, a, s) for a in (*ARMS, *RULES) for s in SCHEDULES},
    }
    for k in ("P1_held_out", "P1_long_gaps", "P_trained", "manipulation_check"):
        res[k]["reading"] = reading(res[k])
    res["usable"] = res["manipulation_check"]["ci95"][0] > 0
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, s) for a in (*RULES, *ARMS) for w in w4.WEIGHTS for s in SEEDS]
    jobs.sort(key=lambda j: j[0] != "ode_L")  # the slow arm first, so workers stay busy
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    res = analyse(cells)
    (g.OUT / "w9_pool.json").write_text(json.dumps(res, indent=2))
    print(f"arm valid {res['arm_valid']}  nonfinite {res['nonfinite']}")
    for k in ("manipulation_check", "P1_held_out", "P1_long_gaps", "P_trained", "ode_vs_lstm_held_out"):
        print(f"  {k:22s} {ci(res[k])}")
    for s, v in res["ode_vs_lstm"].items():
        print(f"  ode_L-lstm_L {s}        {ci(v)}")
    for k, v in res["vs_rules"].items():
        print(f"  {k:22s} {ci(v)}")
    print(f"GATE usable={res['usable']} ({res['manipulation_check']['reading']})   "
          f"P1 (held out): {res['P1_held_out']['reading']}   long gaps: {res['P1_long_gaps']['reading']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        (SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(worker(arch, w, seed)))
    else:
        sys.exit(main())
