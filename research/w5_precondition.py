"""W5-pre: does treated Hahnfeldt / Kuznetsov have TTP dynamic range?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] W5-pre, before any
evaluation ran. Rules only -- M1's policy set on the two non-LV families,
provisional theta ranges, regular and 50%-missed weekly visits.

  Pass (per family, 357 d, p 0), all of:
    P1  MTD capped median TTP >= 28 d
    P2  MTD or best thr: IQR width >= 14 d and censoring in [5%, 95%]
    P3  some thr_x vs MTD: |median paired dTTP| >= 7 d, two-sided Wilcoxon p < 0.01
  and every trajectory finite.

Run: JAX_PLATFORMS=cpu uv run python research/w5_precondition.py
     -> research/outputs/w5_precondition.{json,log}
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_monitoring as m1  # noqa: E402  (imports w4_irregular: backend env first)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import (  # noqa: E402
    burden_integral,
    cumulative_exposure,
    paired_wilcoxon,
    ttp_running_nadir,
)

g = m1.g
FAMILIES = ("hahnfeldt", "kuznetsov")
SEEDS = (1, 2, 3)
N = 200
POLICIES = ("none", "MTD", "AT50", *(f"thr{x}" for x in m1.THRESHOLDS))
HORIZONS = (52, 156)  # weekly visits: 357 d (primary) and 1085 d
DROPS = (0.0, 0.5)
RESULT = g.OUT / "w5_precondition.json"


def evaluate(fam: str, n_visits: int, p: float) -> dict:
    """{policy: {field: (n_seed * N,)}} pooled over seeds, paired by patient."""
    ep = replace(g.EP, model_id=fam, n_visits=n_visits, drop_rate=p, miss_dose="hold")
    ts = np.arange(n_visits) * ep.visit_interval
    horizon = float(ts[-1])
    out = {k: {f: [] for f in ("ttp", "censored", "exposure", "burden_integral")} for k in POLICIES}
    for seed in SEEDS:
        cseed = 300_000 + seed
        c = sample_cohort(fam, N, seed=cseed)
        th, y0, key = jnp.asarray(c.thetas), jnp.asarray(c.y0s), jax.random.PRNGKey(cseed)
        for name in POLICIES:
            burden, dose = (np.asarray(x) for x in m1._rollout(m1.policy(name), th, y0, key, ep))
            ttp = ttp_running_nadir(burden, ts)  # raises on any non-finite trajectory
            out[name]["ttp"].append(np.minimum(ttp.days, horizon))
            out[name]["censored"].append(ttp.censored)
            out[name]["exposure"].append(cumulative_exposure(dose, ts))
            # relative to baseline, so the two families' scales are comparable
            out[name]["burden_integral"].append(burden_integral(burden / burden[:, :1], ts))
    return {k: {f: np.concatenate(v) for f, v in d.items()} for k, d in out.items()}


def summary(a: dict) -> dict:
    ttp = a["ttp"]
    q1, q3 = np.percentile(ttp, [25, 75])
    return {
        "ttp_median": float(np.median(ttp)),
        "ttp_iqr": [float(q1), float(q3)],
        "censoring": float(a["censored"].mean()),
        "exposure_median": float(np.median(a["exposure"])),
        "burden_integral_rel_median": float(np.median(a["burden_integral"])),
    }


def gate(r: dict) -> dict:
    """The registered pass criteria on one family's (357 d, p 0) results."""
    s = {k: summary(v) for k, v in r.items()}
    p1 = s["MTD"]["ttp_median"] >= 28.0

    def spread(k):
        lo, hi = s[k]["ttp_iqr"]
        return hi - lo >= 14.0 and 0.05 <= s[k]["censoring"] <= 0.95

    thr = [k for k in POLICIES if k.startswith("thr")]
    best = max(thr, key=lambda k: s[k]["ttp_median"])
    p2 = spread("MTD") or spread(best)
    sep = {}
    for k in thr:
        d = r[k]["ttp"] - r["MTD"]["ttp"]
        w = paired_wilcoxon(r[k]["ttp"], r["MTD"]["ttp"]) if np.any(d != 0) else {"p_value": 1.0}
        sep[k] = {"median_paired_dttp": float(np.median(d)), "p": float(w["p_value"])}
    p3 = any(abs(v["median_paired_dttp"]) >= 7.0 and v["p"] < 0.01 for v in sep.values())
    return {
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "pass": bool(p1 and p2 and p3),
        "best_thr": best,
        "thr_vs_mtd": sep,
    }


def main() -> int:
    res = {"prereg": "adaptive_therapy/PREREG.md W5-pre [2026-09-30]", "families": {}}
    for fam in FAMILIES:
        rf = {}
        for nv in HORIZONS:
            for p in DROPS:
                r = evaluate(fam, nv, p)
                cell = {k: summary(v) for k, v in r.items()}
                if nv == 52 and p == 0.0:
                    rf["gate"] = gate(r)
                rf[f"{nv * 7 - 7}d_p{p}"] = cell
                print(f"\n{fam}  horizon {nv * 7 - 7} d  p {p}")
                print(
                    f"  {'policy':8s} {'TTP med [IQR]':>22s} {'cens':>6s} {'expo':>7s} {'AUC/b0':>9s}"
                )
                for k, v in cell.items():
                    lo, hi = v["ttp_iqr"]
                    print(
                        f"  {k:8s} {v['ttp_median']:7.1f} [{lo:6.1f},{hi:6.1f}] "
                        f"{v['censoring']:6.2f} {v['exposure_median']:7.1f} {v['burden_integral_rel_median']:9.1f}"
                    )
        gt = rf["gate"]
        print(f"\n{fam} GATE: P1={gt['P1']} P2={gt['P2']} P3={gt['P3']} -> PASS={gt['pass']}")
        for k, v in gt["thr_vs_mtd"].items():
            print(f"    {k} - MTD: median dTTP {v['median_paired_dttp']:+.1f} d, p={v['p']:.2g}")
        res["families"][fam] = rf
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
