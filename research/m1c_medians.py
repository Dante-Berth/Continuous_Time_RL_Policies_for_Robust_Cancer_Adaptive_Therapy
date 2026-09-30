"""M1c: M1/M1b's monitoring map with medians in place of means.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] M1c, post-hoc on the
saved M1/M1b shards (no new simulation). Best x is chosen per (seed, bin, p,
miss rule, w) by MEDIAN U on the selection cohort; Delta_med = median U(best
thr) - median U(MTD) over pooled test patients, mean over w.

  P1  M1b, p 0.5, hold, 4 high-turnover Strobl bins: CI > 0 in all 4 => the
      three-year reversal holds for the median patient; CI < 0 or covering 0 in
      all 4 => a mean/tail effect; else mixed.
  P2  H_cost under Delta_med at both horizons: CI < 0 in both turnover bins.

Run: JAX_PLATFORMS=cpu uv run python research/m1c_medians.py
     -> research/outputs/m1c_medians.{json,log}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_monitoring as m1  # noqa: E402  (imports w4_irregular: backend env first)

import json  # noqa: E402

import numpy as np  # noqa: E402

g, w4 = m1.g, m1.w4
HORIZONS = {"M1": (m1.SHARDS, w4.HORIZON), "M1b": (g.OUT / "m1b_shards", 1085.0)}
HIGH_DT = [b for b in m1.STROBL_BINS if b >= 5]
RESULT = g.OUT / "m1c_medians.json"


def load(shards: Path) -> dict:
    sh = [np.load(shards / f"seed{s}.npz") for s in m1.SEEDS]
    return {f: np.stack([x[f] for x in sh]) for f in m1.FIELDS}


def arms(a: dict, b: int, mi: int, pi: int, agg=np.median) -> tuple[np.ndarray, ...]:
    """(ub, um, best): test U of the selection-chosen thr and of MTD, (n_w, n_seed, N);
    best (n_w, n_seed) policy index, chosen by `agg` U on the selection cohort."""
    thr = np.asarray([m1.POLICIES.index(f"thr{x}") for x in m1.THRESHOLDS])
    mtd = m1.POLICIES.index("MTD")
    ub, um, best = [], [], []
    for w in w4.WEIGHTS:
        u = m1.utility(a, w)
        k = thr[agg(u[:, b, 0, mi, pi][:, thr], axis=-1).argmax(-1)]
        test = u[:, b, 1, mi, pi]
        ub.append(test[np.arange(len(m1.SEEDS)), k])
        um.append(test[:, mtd])
        best.append(k)
    return np.stack(ub), np.stack(um), np.stack(best)


def delta_med(ub: np.ndarray, um: np.ndarray) -> float:
    n_w = ub.shape[0]
    return float(np.mean(np.median(ub.reshape(n_w, -1), -1) - np.median(um.reshape(n_w, -1), -1)))


def draws(ub: np.ndarray, um: np.ndarray, seed: int) -> np.ndarray:
    """Seed+patient bootstrap, both arms resampled jointly, as w4.delta_draws."""
    rng = np.random.default_rng(seed)
    pi = rng.integers(0, g.N_TEST, (w4.N_BOOT, g.N_TEST))
    si = rng.integers(0, ub.shape[1], (w4.N_BOOT, ub.shape[1]))
    return np.array([delta_med(ub[:, s][..., i], um[:, s][..., i]) for s, i in zip(si, pi)])


def standing(a: dict, b: int, best: np.ndarray) -> dict:
    """Capped median TTP + IQR, censoring, median exposure at p 0.5, hold, w 0.1."""
    mi, pi, wi = m1.MISS.index("hold"), m1.P_TEST, w4.WEIGHTS.index(0.1)
    rows = {"best_thr": best[wi], "MTD": [m1.POLICIES.index("MTD")] * len(m1.SEEDS)}
    out = {}
    for name, ks in rows.items():
        pool = {
            f: np.concatenate([a[f][s, b, 1, mi, pi, k] for s, k in enumerate(ks)])
            for f in m1.FIELDS
        }
        cap = np.minimum(pool["ttp"], m1.HORIZON)
        out[name] = {
            "ttp_median": float(np.median(cap)),
            "ttp_iqr": np.percentile(cap, [25, 75]).tolist(),
            "censoring": float(pool["censored"].mean()),
            "exposure_median": float(np.median(pool["exposure"])),
            "burden_integral_median": float(np.median(pool["burden_integral"])),
        }
    u = m1.utility(a, 0.1)[:, b, 1, mi, pi]
    ub = np.concatenate([u[s, k] for s, k in enumerate(best[wi])])
    um = u[:, m1.POLICIES.index("MTD")].ravel()
    out["wilcoxon_best_gt_mtd_p"] = m1.paired_wilcoxon(ub, um, alternative="greater")["p_value"]
    out["chosen_x"] = [m1.POLICIES[k] for k in best[wi]]
    return out


def horizon(a: dict) -> dict:
    res, at = {"bins": {}}, {}
    for b in range(len(m1.BINS)):
        rb = {"outside_strobl_sweep": b not in m1.STROBL_BINS}
        for mi, miss in enumerate(m1.MISS):
            deltas = []
            for pi in range(len(m1.DROPS)):
                ub, um, best = arms(a, b, mi, pi)
                deltas.append(delta_med(ub, um))
                if miss == "hold" and pi == m1.P_TEST:
                    at[b] = (ub, um)
                    dr = draws(ub, um, seed=b)
                    rb["p0.5_ci95"] = np.percentile(dr, [2.5, 97.5]).tolist()
                    rb["p0.5_standing"] = standing(a, b, best)
                    mb, mm, _ = arms(a, b, mi, pi, agg=np.mean)
                    rb["p0.5_delta_med_mean_selected"] = delta_med(mb, mm)
            rb[miss] = {"delta_med": deltas, "p_star": m1.crossing(deltas)}
        res["bins"][m1.label(b)] = rb
    res["H_cost"] = {}
    for ti in range(len(m1.DT)):
        hi, lo = 5 * ti + 1, 5 * ti + 4
        d = draws(*at[hi], seed=hi) - draws(*at[lo], seed=lo)
        res["H_cost"][f"dT{m1.DT[ti]}"] = {
            "high_cost_minus_no_cost": delta_med(*at[hi]) - delta_med(*at[lo]),
            "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
        }
    res["H_cost_median_holds"] = all(v["ci95"][1] < 0 for v in res["H_cost"].values())
    return res


def p1_reading(res: dict) -> str:
    cis = [res["bins"][m1.label(b)]["p0.5_ci95"] for b in HIGH_DT]
    if all(lo > 0 for lo, _ in cis):
        return "the three-year reversal holds for the median patient"
    if all(lo <= 0 for lo, _ in cis):
        return "the reversal is a mean/tail effect"
    return "mixed"


def main() -> int:
    out = {"prereg": "adaptive_therapy/PREREG.md M1c [2026-09-30]", "post_hoc": True}
    for name, (shards, h) in HORIZONS.items():
        m1.HORIZON = h
        out[name] = horizon(load(shards))
        out[name]["horizon_days"] = h
        print(f"== {name} ({h:.0f} d): Delta_med, hold | p*_med hold, full")
        for lab, rb in out[name]["bins"].items():
            row = " ".join(f"{v:+6.1f}" for v in rb["hold"]["delta_med"])
            lo, hi = rb["p0.5_ci95"]
            st = rb["p0.5_standing"]
            print(
                f"  {lab:28s} {row}  {rb['hold']['p_star']!s:>8} {rb['full']['p_star']!s:>8}"
                f"  p.5 CI [{lo:+.0f},{hi:+.0f}] meanSel {rb['p0.5_delta_med_mean_selected']:+.0f}"
                f"  TTP thr {st['best_thr']['ttp_median']:.0f} vs MTD {st['MTD']['ttp_median']:.0f}"
                f"  exp {st['best_thr']['exposure_median']:.0f} vs {st['MTD']['exposure_median']:.0f}"
            )
        for k, v in out[name]["H_cost"].items():
            lo, hi = v["ci95"]
            print(f"  H_cost_med {k}: {v['high_cost_minus_no_cost']:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    out["P1_reading"] = p1_reading(out["M1b"])
    print(f"P1: {out['P1_reading']}")
    print("P2: " + ", ".join(f"{n} {out[n]['H_cost_median_holds']}" for n in HORIZONS))
    RESULT.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
