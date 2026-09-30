"""M1: when does monitoring stop paying, across cost of resistance and turnover?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] M1, before any M1
evaluation ran. Rules only -- no learned policy, no architecture.

  Bins      r_R/r_S {[.25,.5) outside Strobl's sweep, [.5,.625), [.625,.75),
            [.75,.875), [.875,1]} x d_T/r_S {[0,.25), [.25,.5]}, Strobl ranges
            otherwise; per seed and bin a 200-patient SELECTION cohort and a
            200-patient TEST cohort.
  Policies  MTD; thr_x (full dose above x * baseline), x in .5..1.2; AT50.
  Visits    weekly, missed i.i.d. with p in 0..0.6; miss rule hold | full.
  Primary   Delta_fb = U(best thr) - U(MTD) on TEST patients, best x chosen on
            the selection cohort per (seed, bin, p, rule, w); H_cost: at p 0.5
            under hold, highest-cost Strobl bin minus no-cost-end bin, CI
            excluding 0 in both turnover bins.

Run: JAX_PLATFORMS=cpu uv run python research/m1_monitoring.py
     -> research/outputs/m1_monitoring.{json,log}, m1_shards/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w4_irregular as w4  # noqa: E402  (imports phase2_gate: backend env first)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import (  # noqa: E402
    burden_integral,
    cumulative_exposure,
    paired_wilcoxon,
    ttp_running_nadir,
)
from adaptive_therapy.policies.rules import RulePolicy, ThresholdPolicy  # noqa: E402
from adaptive_therapy.rl import ppo  # noqa: E402

g, wf = w4.g, w4.wf
RR = ((0.25, 0.5), (0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0))
DT = ((0.0, 0.25), (0.25, 0.5))
BINS = [(rr, dt) for dt in DT for rr in RR]  # index = 5 * dt_i + rr_i
STROBL_BINS = [i for i, (rr, _) in enumerate(BINS) if rr[0] >= 0.5]
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
POLICIES = ("MTD", "AT50", *(f"thr{x}" for x in THRESHOLDS))
DROPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
MISS = ("hold", "full")
COHORTS = ("sel", "test")
SEEDS = tuple(range(1, 11))
N = g.N_TEST  # 200 patients per cohort
P_TEST = DROPS.index(0.5)
SHARDS = g.OUT / "m1_shards"
# Episode settings, rebound by research/m1b_horizon.py (a longer horizon); every
# function below reads these, never g.EP / g.TS / w4.HORIZON directly.
EP, TS, HORIZON = g.EP, g.TS, w4.HORIZON
SCRIPT = Path(__file__)  # the file a worker process runs
RESULT = g.OUT / "m1_monitoring.json"
FIELDS = ("ttp", "censored", "exposure", "burden_integral")


def label(b: int) -> str:
    (r0, r1), (t0, t1) = BINS[b]
    return f"rR[{r0},{r1})|dT[{t0},{t1})"


def policy(name: str):
    return ThresholdPolicy(float(name[3:])) if name.startswith("thr") else RulePolicy(name)


@eqx.filter_jit
def _rollout(pol, th, y0, key, ep):
    ev = ppo.evaluate(pol, th, y0, key, ep, RewardConfig())
    return ev["burden"], ev["dose"]


def worker(seed: int) -> None:
    """All bins x cohorts x rules x rates x policies for one seed -> one .npz."""
    out = {f: np.zeros((len(BINS), 2, 2, len(DROPS), len(POLICIES), N), np.float32) for f in FIELDS}
    for b, (rr, dt) in enumerate(BINS):
        for ci, _ in enumerate(COHORTS):
            cseed = (ci + 1) * 100_000 + 1000 * b + seed
            c = sample_cohort(
                g.FAM, N, seed=cseed, ranges="strobl2021", bounds={"r_R/r_S": rr, "d_T/r_S": dt}
            )
            th, y0, key = jnp.asarray(c.thetas), jnp.asarray(c.y0s), jax.random.PRNGKey(cseed)
            for mi, miss in enumerate(MISS):
                for pi, p in enumerate(DROPS):
                    ep = replace(EP, drop_rate=p, miss_dose=miss)
                    for k, name in enumerate(POLICIES):
                        burden, dose = (
                            np.asarray(x) for x in _rollout(policy(name), th, y0, key, ep)
                        )
                        ttp = ttp_running_nadir(burden, TS)
                        out["ttp"][b, ci, mi, pi, k] = ttp.days
                        out["censored"][b, ci, mi, pi, k] = ttp.censored
                        out["exposure"][b, ci, mi, pi, k] = cumulative_exposure(dose, TS)
                        out["burden_integral"][b, ci, mi, pi, k] = burden_integral(burden, TS)
    np.savez_compressed(SHARDS / f"seed{seed}.npz", **out)


def spawn(seed: int) -> None:
    if not (SHARDS / f"seed{seed}.npz").exists():  # resume: a finished seed is never recomputed
        subprocess.run(
            [sys.executable, str(SCRIPT), "--worker", str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done seed {seed}", flush=True)


# ---------------------------------------------------------------- analysis


def utility(a: dict, w: float) -> np.ndarray:
    return np.minimum(a["ttp"], HORIZON) - w * a["exposure"]


def paired(a: dict, b: int, mi: int, pi: int) -> tuple[np.ndarray, np.ndarray]:
    """(d, choice): d (n_w, 1, n_seed, N) = U(best thr) - U(MTD) on TEST patients,
    best x chosen on the SELECTION cohort; choice (n_w, n_seed) = chosen x."""
    thr = [POLICIES.index(f"thr{x}") for x in THRESHOLDS]
    mtd = POLICIES.index("MTD")
    d = np.zeros((len(w4.WEIGHTS), 1, len(SEEDS), N))
    choice = np.zeros((len(w4.WEIGHTS), len(SEEDS)))
    for wi, w in enumerate(w4.WEIGHTS):
        u = utility(a, w)  # (seed, bin, cohort, miss, p, policy, patient)
        sel = u[:, b, 0, mi, pi][:, thr].mean(-1)  # (seed, n_thr)
        best = np.asarray(thr)[sel.argmax(-1)]
        test = u[:, b, 1, mi, pi]
        d[wi, 0] = test[np.arange(len(SEEDS)), best] - test[:, mtd]
        choice[wi] = np.asarray(THRESHOLDS)[sel.argmax(-1)]
    return d, choice


def best_u(a: dict, b: int, mi: int, pi: int) -> np.ndarray:
    """(n_w, n_seed, N) test U of the selection-chosen threshold."""
    d, _ = paired(a, b, mi, pi)
    mtd = POLICIES.index("MTD")
    base = np.stack([utility(a, w)[:, b, 1, mi, pi, mtd] for w in w4.WEIGHTS])
    return d[:, 0] + base


def crossing(deltas: list[float]) -> str | float:
    """First p where the Delta_fb point estimate crosses 0, linearly interpolated."""
    if deltas[0] <= 0:
        return "<=0 at p 0"
    for (p0, d0), (p1, d1) in zip(zip(DROPS, deltas), zip(DROPS[1:], deltas[1:])):
        if d1 <= 0 < d0:
            return round(p0 + (p1 - p0) * d0 / (d0 - d1), 3)
    return f">{DROPS[-1]}"


def describe(a: dict, b: int, choice_w01: np.ndarray) -> dict:
    """Standing criteria at p 0.5 under hold, w 0.1: best thr (per-seed choice) vs MTD."""
    mi, pi, wi = MISS.index("hold"), P_TEST, w4.WEIGHTS.index(0.1)
    out = {}
    rows = {
        "best_thr": [POLICIES.index(f"thr{x}") for x in choice_w01[wi]],
        "MTD": [POLICIES.index("MTD")] * len(SEEDS),
        "AT50": [POLICIES.index("AT50")] * len(SEEDS),
    }
    for name, ks in rows.items():
        pool = {
            f: np.concatenate([a[f][s, b, 1, mi, pi, k] for s, k in enumerate(ks)]) for f in FIELDS
        }
        cap = np.minimum(pool["ttp"], HORIZON)
        out[name] = {
            "ttp_median": float(np.median(cap)),
            "ttp_iqr": np.percentile(cap, [25, 75]).tolist(),
            "censoring": float(pool["censored"].mean()),
            "exposure_median": float(np.median(pool["exposure"])),
            "burden_integral_median": float(np.median(pool["burden_integral"])),
        }
    ub = np.concatenate(
        [utility(a, 0.1)[s, b, 1, mi, pi, k] for s, k in enumerate(rows["best_thr"])]
    )
    um = utility(a, 0.1)[:, b, 1, mi, pi, POLICIES.index("MTD")].ravel()
    out["wilcoxon_best_gt_mtd_p"] = paired_wilcoxon(ub, um, alternative="greater")["p_value"]
    return out


def analyse() -> dict:
    shards = [np.load(SHARDS / f"seed{s}.npz") for s in SEEDS]
    a = {f: np.stack([sh[f] for sh in shards]) for f in FIELDS}
    res = {"prereg": "adaptive_therapy/PREREG.md M1 [2026-09-30]", "bins": {}}
    d_at = {}
    for b in range(len(BINS)):
        rb = {"outside_strobl_sweep": b not in STROBL_BINS}
        for mi, miss in enumerate(MISS):
            deltas, cis, choices = [], [], []
            for pi, p in enumerate(DROPS):
                d, choice = paired(a, b, mi, pi)
                ci = w4.delta_ci(d, (p,), seed=b)
                deltas.append(ci["delta"])
                cis.append(ci["ci95"])
                choices.append(choice[w4.WEIGHTS.index(0.1)].tolist())
                d_at[(b, mi, pi)] = d
            rb[miss] = {
                "delta": deltas,
                "ci95": cis,
                "p_star": crossing(deltas),
                "choice_w01": choices,
            }
        _, ch = paired(a, b, MISS.index("hold"), P_TEST)
        rb["p0.5_hold_standing"] = describe(a, b, ch)
        res["bins"][label(b)] = rb
    # H_cost: highest-cost Strobl bin minus no-cost-end bin, p 0.5, hold, per turnover bin
    hold = MISS.index("hold")
    res["H_cost"] = {}
    for ti in range(len(DT)):
        hi, lo = 5 * ti + 1, 5 * ti + 4
        dh, dl = d_at[(hi, hold, P_TEST)], d_at[(lo, hold, P_TEST)]
        draws = w4.delta_draws(dh, seed=hi)[0] - w4.delta_draws(dl, seed=lo)[0]
        point = float(np.nanmean(dh) - np.nanmean(dl))
        res["H_cost"][f"dT{DT[ti]}"] = {
            "high_cost_minus_no_cost": point,
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        }
    cis = [v["ci95"] for v in res["H_cost"].values()]
    res["H_cost_passes"] = bool(all(lo > 0 or hi < 0 for lo, hi in cis))
    # H_miss: best thr under full minus under hold, p 0.5, pooled over Strobl bins
    full = MISS.index("full")
    dm = np.stack(
        [best_u(a, b, full, P_TEST) - best_u(a, b, hold, P_TEST) for b in STROBL_BINS], axis=1
    )  # (n_w, n_bins, n_seed, N)
    res["H_miss"] = w4.delta_ci(dm, tuple(STROBL_BINS))
    lo, hi = res["H_miss"]["ci95"]
    res["H_miss_reading"] = (
        "hold-protocol artefact"
        if lo > 0
        else "dosing through a gap costs more"
        if hi < 0
        else "inconclusive"
    )
    return res


def report(res: dict) -> None:
    print("Delta_fb = U(best thr) - U(MTD), test patients, mean over w; hold | full; p* = crossing")
    print(f"{'bin':28s} " + " ".join(f"p{p:<5}" for p in DROPS) + "  p*(hold) p*(full)")
    for name, rb in res["bins"].items():
        row = " ".join(f"{v:+6.1f}" for v in rb["hold"]["delta"])
        tag = " (outside sweep)" if rb["outside_strobl_sweep"] else ""
        print(f"{name:28s} {row}  {rb['hold']['p_star']!s:>8} {rb['full']['p_star']!s:>8}{tag}")
    for k, v in res["H_cost"].items():
        print(
            f"H_cost {k}: {v['high_cost_minus_no_cost']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
        )
    print(f"H_cost PASS={res['H_cost_passes']}")
    m = res["H_miss"]
    print(
        f"H_miss (full - hold, p 0.5): {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}] -> {res['H_miss_reading']}"
    )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        list(ex.map(spawn, SEEDS))
    res = analyse()
    report(res)
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
        raise SystemExit(0)
    raise SystemExit(main())
