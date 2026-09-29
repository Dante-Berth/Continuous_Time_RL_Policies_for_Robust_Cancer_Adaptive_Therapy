"""W4b: W4 re-registered intention-to-treat, with MTD as a reference arm.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-29] W4b, before any W4b
cell ran. Everything W4 fixed (missed-visit hold, arms, PQN, p_train 0.35, w
grid, test drops, U, seed+patient bootstrap) is imported from w4_irregular.py;
what changes is below.

  Seeds     fresh {21..30}: W4's cells were seen, so they cannot be re-graded.
  Validity  intention-to-treat. W4's closed-loop gate excluded near-fixed
            schedules that scored HIGHER than valid cells (RESEARCH_LOG L632);
            at Tier 4 a fixed schedule is a policy, not a collapse. A cell is
            excluded only if non-finite; an arm with > 5% such cells is INVALID.
  H_ct4b    (primary) ODE-RNN CI above 0 against EVERY X in {MLP, LSTM,
            LSTM+dt, MTD}, mean over w and p in {.2, .5}.
  H_grace   (secondary) the same on U(p) - U(0), X in {MLP, LSTM, LSTM+dt}.
  H_fb      (secondary) best learned arm by point Delta vs MTD at p = .5 only.
  Post-hoc  W4's own cells under the ITT rule, labelled as not a test.

Run: JAX_PLATFORMS=cpu uv run python research/w4b_itt.py
     -> research/outputs/w4b_itt.{json,png}, w4b_shards/, w4b_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w4_irregular as w4  # noqa: E402  (imports phase2_gate: backend env first)

import numpy as np  # noqa: E402

from adaptive_therapy.policies.rules import RULES  # noqa: E402

g, wf = w4.g, w4.wf
SEEDS = tuple(range(21, 31))
LEARNED = ("mlp", "lstm", "lstm_dt")
MAX_NONFINITE = 0.05
SHARDS = g.OUT / "w4b_shards"
POLICIES = g.OUT / "w4b_policies"


def finite(cell: dict) -> bool:
    return all(
        np.isfinite(cell[str(p)][k]).all() for p in w4.DROPS for k in ("ttp_capped", "exposure")
    )


def itt_validity(cells, archs, seeds) -> tuple[dict, dict]:
    """Every finite cell is valid; an arm is INVALID above 5% non-finite cells."""
    valid = {(c["arch"], c["w"], c["seed"]): finite(c) for c in cells}
    n = len(w4.WEIGHTS) * len(seeds)
    arm_valid = {
        a: sum(not valid[(a, w, s)] for w in w4.WEIGHTS for s in seeds) / n <= MAX_NONFINITE
        for a in archs
    }
    return valid, arm_valid


def with_mtd(cells, rules) -> list[dict]:
    """MTD as a per-w pseudo-arm: its trajectory is w-free, U is formed per w."""
    mtd = [{**r, "w": w} for r in rules if r["arch"] == "MTD" for w in w4.WEIGHTS]
    return cells + mtd


def analyse(cells, rules, seeds) -> dict:
    cells = with_mtd(cells, rules)
    valid, arm_valid = itt_validity(cells, (*w4.ARCHS, "MTD"), seeds)
    res = {
        "arm_valid": arm_valid,
        "nonfinite": [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok],
        "table": w4.table(cells, valid, rules, tuple(w4.ARCHS), seeds),
        "closed_loop": {a: [c["closed_loop"] for c in cells if c["arch"] == a] for a in w4.ARCHS},
    }

    def cmp(a, b, stat, drops):
        return w4.comparison(cells, valid, a, b, stat, seeds, drops)

    for h, stat, others in (
        ("H_ct4b", "level", (*LEARNED, "MTD")),
        ("H_grace", "degradation", LEARNED),
    ):
        r = {"valid": all(arm_valid[x] for x in ("ode_rnn", *others)), "vs": {}}
        for x in others:
            r["vs"][x] = cmp("ode_rnn", x, stat, w4.TEST_DROPS)
            if stat == "level":
                r["vs"][x]["drop0"] = cmp("ode_rnn", x, stat, (0.0,))
                r["vs"][x]["wilcoxon_greater_p"] = w4.wilcoxon_per_w_p(
                    cells, valid, "ode_rnn", x, seeds
                )
        r["passes"] = bool(r["valid"] and min(v["ci95"][0] for v in r["vs"].values()) > 0)
        res[h] = r
    fb = {a: cmp(a, "MTD", "level", (0.5,)) for a in w4.ARCHS}
    best = max(fb, key=lambda a: fb[a]["delta"])
    res["H_fb"] = {"best": best, "vs_mtd_p0.5": fb, "passes": fb[best]["ci95"][0] > 0}
    res["dt_input_effect"] = cmp("lstm_dt", "lstm", "level", w4.TEST_DROPS)
    return res


def spawn(kind: str, arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", kind, arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {kind:5s} {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def report(res: dict) -> None:
    for h in ("H_ct4b", "H_grace"):
        r = res[h]
        print(
            f"{h}: "
            + "  ".join(
                f"ODE−{x} {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                for x, v in r["vs"].items()
            )
            + f"  valid={r['valid']} PASS={r['passes']}"
        )
    fb = res["H_fb"]
    v = fb["vs_mtd_p0.5"][fb["best"]]
    print(
        f"H_fb: {fb['best']}−MTD @p0.5 {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
        f" PASS={fb['passes']}"
    )
    d = res["dt_input_effect"]
    print(f"LSTM+dt − LSTM {d['delta']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
    print(f"arm valid {res['arm_valid']}  nonfinite {res['nonfinite']}")
    for a, cl in res["closed_loop"].items():
        print(
            f"  closed-loop {a:8s} median {np.median(cl):.3f}  <0.10: {np.mean(np.array(cl) < 0.10):.0%}"
        )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [("train", a, w, s) for a in w4.ARCHS for w in w4.WEIGHTS for s in SEEDS] + [
        ("rule", r, 0.0, s) for r in RULES for s in SEEDS
    ]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        out = list(ex.map(lambda j: spawn(*j), jobs))
    train = [c for j, c in zip(jobs, out, strict=True) if j[0] == "train"]
    rules = [c for j, c in zip(jobs, out, strict=True) if j[0] == "rule"]
    res = {"prereg": "adaptive_therapy/PREREG.md W4b [2026-09-29]", **analyse(train, rules, SEEDS)}
    print("== W4b (registered) =="), report(res)
    # post-hoc, NOT a test: W4's seen cells under the ITT rule
    old = [json.loads(p.read_text()) for p in sorted(w4.SHARDS.glob("*.json"))]
    res["posthoc_w4_itt"] = analyse(
        [c for c in old if c["arch"] in w4.ARCHS and "closed_loop" in c and c["seed"] in w4.SEEDS],
        [c for c in old if c["arch"] in RULES and c["seed"] in w4.SEEDS],
        w4.SEEDS,
    )
    print("== W4 cells under ITT (post-hoc, not a test) =="), report(res["posthoc_w4_itt"])
    (g.OUT / "w4b_itt.json").write_text(json.dumps(res, indent=2))
    rows = (("b", "W4b seeds 21–30 (registered)"), ("a", "W4 seeds 11–20 (post-hoc ITT)"))
    tabs = {"b_table": res["table"], "a_table": res["posthoc_w4_itt"]["table"]}
    w4.plot(tabs, g.OUT / "w4b_itt.png", rows, "W4b")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        kind, arch, w, seed = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        cell = (
            w4.train_cell(arch, w, seed, POLICIES) if kind == "train" else w4.rules_cell(arch, seed)
        )
        (SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json").write_text(json.dumps(cell))
        raise SystemExit(0)
    raise SystemExit(main())
