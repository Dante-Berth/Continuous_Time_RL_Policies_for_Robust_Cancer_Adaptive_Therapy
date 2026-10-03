"""V6-gate: fit the Baez-Kuang/Phan two-population Droop model to our Bruchovsky patients.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-04] V6-gate, before any fit ran.
Protocol: Phan et al. 2019 p.189 (cohort), p.193 (objective), Tables 2-3 p.197-199
(literature/papers/doi-10.3934-mbe.2019010.md, 1 deg).

  Fit       MSE_PSA + MSE_androgen over phases 0-2 (on, off, on), Table 1 bounds,
            30 log-uniform restarts (declared deviation: Phan's guesses were by hand).
  Forecast  phases 3-4 (the next off/on cycle), u from the data, same trajectory.
  Pass      on the 27-patient cohort, median fit RMSE PSA <= 1.43 and androgen
            <= 3.35; median forecast RMSE PSA <= 6.78 and androgen <= 6.29.

Data (gitignored, never committed): data/bruchovsky/dataTanaka/.

Run: JAX_PLATFORMS=cpu uv run python research/v6_phan_fit.py
     -> research/outputs/v6_phan_fit.{json,log}
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import csv
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from adaptive_therapy.envs import droop
from adaptive_therapy.eval.metrics import ttp_running_nadir

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bruchovsky" / "dataTanaka" / "Bruchovsky_et_al"
OUT = Path(__file__).resolve().parent / "outputs" / "v6_phan_fit.json"
# 20 steps/day: off treatment gamma ~ 20/day, so h = 0.05 keeps RK4 at h*gamma = 1,
# inside its stability region (2.8); 10 steps/day would sit at 2.0.
STEPS_PER_DAY = 20
N_DAYS = 3000  # one compiled length for every patient; longer histories are cut here
N_RESTARTS = 30
MIN_POINTS = 20  # p.189
SEED = 0
LO = np.log(np.array(droop.LO + (1e-2, 1e-2)))  # + x1(0), x2(0) in mm^3 (PREREG)
HI = np.log(np.array(droop.HI + (1e4, 1e4)))
NAMES = droop.THETA_NAMES + ("x1_0", "x2_0")
GATE = {"fit_psa": 1.43, "fit_and": 3.35, "fc_psa": 6.78, "fc_and": 6.29}
CAS_DAYS = 1095


def load(path: Path) -> dict:
    """Rows sorted by day; psa/testosterone NaN where unrecorded; phase index per row."""
    rows = [r for r in csv.reader(path.open()) if len(r) >= 9 and r[8].strip()]
    rows.sort(key=lambda r: float(r[8]))  # 016/037 carry out-of-order rows (R2 amendment)
    days = np.array([float(r[8]) for r in rows])
    f = lambda s: float(s) if s.strip() else np.nan  # noqa: E731
    psa = np.array([f(r[4]) for r in rows])
    tst = np.array([f(r[5]) for r in rows])
    on, last = [], 1.0
    for r in rows:  # an unrecorded flag continues the previous one (as R2)
        last = float(r[7]) if r[7].strip() else last
        on.append(last)
    on = np.array(on)
    phase = np.concatenate([[0], np.cumsum(on[1:] != on[:-1])])
    return {"days": days - days[0], "psa": psa, "tst": tst, "on": on, "phase": phase}


def eligible(p: dict) -> tuple[bool, bool]:
    """(Phan's rule on phases 0-2, and a further full off/on cycle to forecast)."""
    m = p["phase"] <= 2
    rule = min(np.isfinite(p["psa"][m]).sum(), np.isfinite(p["tst"][m]).sum()) >= MIN_POINTS
    return bool(rule), bool(p["phase"].max() >= 4)


def daily_dose(p: dict) -> np.ndarray:
    grid = np.arange(N_DAYS)
    idx = np.clip(np.searchsorted(p["days"], grid, side="right") - 1, 0, None)
    return p["on"][idx]


def simulate(z, y_q0p0, u_daily):
    """Daily (x1, x2, Q, P) from log-parameters z, fixed-step RK4."""
    x = jnp.exp(z)
    theta, y0 = x[: droop.N_THETA], jnp.concatenate([x[droop.N_THETA :], y_q0p0])
    h = 1.0 / STEPS_PER_DAY

    def day(y, u):
        def sub(w, _):
            f = lambda v: droop.vector_field(0.0, v, (theta, u))  # noqa: E731
            k1 = f(w)
            k2 = f(w + h / 2 * k1)
            k3 = f(w + h / 2 * k2)
            k4 = f(w + h * k3)
            return w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

        y2, _ = jax.lax.scan(sub, y, None, length=STEPS_PER_DAY)
        return y2, y

    _, ys = jax.lax.scan(day, y0, u_daily)
    return ys


def _residuals(z, y_q0p0, u, at, obs, w):
    """Weighted residuals; sum of squares = MSE_PSA + MSE_androgen (p.193)."""
    ys = simulate(z, y_q0p0, u)
    pred = jnp.stack([ys[at[0], 3], ys[at[1], 2]])
    return ((pred - obs) * w).ravel()


RES = jax.jit(_residuals)
JAC = jax.jit(jax.jacfwd(_residuals))
SIM = jax.jit(simulate)


def _series(p: dict, col: str, mask: np.ndarray):
    ok = np.isfinite(p[col]) & mask & (p["days"] < N_DAYS)
    return p["days"][ok].astype(int), p[col][ok]


def _padded(p: dict, mask: np.ndarray, size: int):
    """(at, obs, w) for PSA and androgen, padded to `size` with zero weight."""
    at, obs, w = np.zeros((2, size), int), np.zeros((2, size)), np.zeros((2, size))
    for i, col in enumerate(("psa", "tst")):
        d, v = _series(p, col, mask)
        at[i, : len(d)], obs[i, : len(d)], w[i, : len(d)] = d, v, 1.0 / np.sqrt(max(len(d), 1))
    return jnp.asarray(at), jnp.asarray(obs), jnp.asarray(w)


def rmse(ys: np.ndarray, p: dict, mask: np.ndarray) -> tuple[float, float]:
    out = []
    for col, k in (("psa", 3), ("tst", 2)):
        d, v = _series(p, col, mask)
        out.append(float(np.sqrt(np.mean((ys[d, k] - v) ** 2))) if len(d) else float("nan"))
    return out[0], out[1]


def fit(pid: int) -> dict:
    p = load(DATA / f"patient{pid:03d}.txt")
    rule, fc = eligible(p)
    fit_m, fc_m = p["phase"] <= 2, (p["phase"] >= 3) & (p["phase"] <= 4)
    u = jnp.asarray(daily_dose(p))
    at, obs, w = _padded(p, fit_m, 128)
    first = lambda c: p[c][np.isfinite(p[c])][0]  # noqa: E731
    y_q0p0 = jnp.array([first("tst"), first("psa")])
    rng = np.random.default_rng([SEED, pid])
    runs = []
    for _ in range(N_RESTARTS):
        z0 = LO + rng.uniform(size=len(LO)) * (HI - LO)
        try:
            r = least_squares(
                lambda z: np.asarray(RES(jnp.asarray(z), y_q0p0, u, at, obs, w)),
                z0,
                jac=lambda z: np.asarray(JAC(jnp.asarray(z), y_q0p0, u, at, obs, w)),
                bounds=(LO, HI), method="trf", x_scale="jac", max_nfev=300,
            )  # fmt: skip
        except ValueError:  # non-finite residual at a restart: skip it, counted below
            continue
        if np.isfinite(r.fun).all():
            runs.append((float(np.sum(r.fun**2)), r.x))
    runs.sort(key=lambda t: t[0])
    obj, best = runs[0]
    ys = np.asarray(SIM(jnp.asarray(best), y_q0p0, u))
    cas = np.asarray(SIM(jnp.asarray(best), y_q0p0, jnp.ones(N_DAYS)))[:CAS_DAYS, 3]
    ttp = ttp_running_nadir(cas[None], np.arange(CAS_DAYS, dtype=float), delta=0.25)
    f_psa, f_and = rmse(ys, p, fit_m)
    c_psa, c_and = rmse(ys, p, fc_m) if fc else (float("nan"),) * 2
    return {
        "pid": pid, "rule": rule, "forecastable": fc, "n_restarts_ok": len(runs),
        "objective": obj, "fit_rmse_psa": f_psa, "fit_rmse_and": f_and,
        "fc_rmse_psa": c_psa, "fc_rmse_and": c_and,
        "cas_ttp": float(ttp.days[0]), "cas_censored": bool(ttp.censored[0]),
        "params": {k: float(v) for k, v in zip(NAMES, np.exp(best), strict=True)},
    }  # fmt: skip


def q(a) -> list[float]:
    return [float(np.nanpercentile(a, s)) for s in (50, 25, 75)]


def main() -> int:
    pids = sorted(int(re.findall(r"\d+", f.stem)[0]) for f in DATA.glob("patient*.txt"))
    pids = [pid for pid in pids if eligible(load(DATA / f"patient{pid:03d}.txt"))[0]]
    with Pool(min(20, len(pids))) as pool:
        fits = []
        for r in pool.imap_unordered(fit, pids):
            fits.append(r)
            print(f"  p{r['pid']:03d} fc={r['forecastable']} fit PSA {r['fit_rmse_psa']:.2f} "
                  f"and {r['fit_rmse_and']:.2f} | fc PSA {r['fc_rmse_psa']:.2f} "
                  f"and {r['fc_rmse_and']:.2f} | CAS TTP {r['cas_ttp']:.0f}", flush=True)  # fmt: skip
    fits.sort(key=lambda r: r["pid"])
    prim = [r for r in fits if r["forecastable"]]
    med = {
        k: q([r[k] for r in prim])
        for k in ("fit_rmse_psa", "fit_rmse_and", "fc_rmse_psa", "fc_rmse_and")
    }
    checks = {
        "fit_psa": med["fit_rmse_psa"][0] <= GATE["fit_psa"],
        "fit_and": med["fit_rmse_and"][0] <= GATE["fit_and"],
        "fc_psa": med["fc_rmse_psa"][0] <= GATE["fc_psa"],
        "fc_and": med["fc_rmse_and"][0] <= GATE["fc_and"],
    }
    cas = np.array([r["cas_ttp"] for r in prim])
    res = {
        "prereg": "adaptive_therapy/PREREG.md V6-gate [2026-10-04]",
        "n_rule": len(fits), "n_primary": len(prim),
        "primary_pids": [r["pid"] for r in prim],
        "median_q1_q3": med, "gate": GATE, "checks": checks, "PASS": all(checks.values()),
        "descriptive": {
            "rule_cohort_fit": {k: q([r[k] for r in fits]) for k in ("fit_rmse_psa", "fit_rmse_and")},
            "cas_ttp_median_q1_q3": q(cas),
            "cas_share_censored": float(np.mean([r["cas_censored"] for r in prim])),
            "phan_examples": {r["pid"]: r for r in fits if r["pid"] in (28, 29, 77, 100)},
        },
        "fits": fits,
    }  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {k: res[k] for k in ("n_rule", "n_primary", "median_q1_q3", "checks", "PASS")}, indent=1
        )
    )
    print(
        json.dumps({k: v for k, v in res["descriptive"].items() if k != "phan_examples"}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
