"""R2: reproduce Strobl 2021's LV fits to the Bruchovsky IADT cohort.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] R2, before any fit ran.
Protocol: Strobl et al. 2021 p.6 (literature/papers/pmc-8455086.md, 1°).

  Fit     PSA(t)/PSA(0) vs n(t)/n0, our LV vector field, K = 1, r_S and d_D
          fixed; r_R/r_S, d_T/r_S, n0, f_R free and bounded; 25 restarts.
  Pass    F1 10-20 patients with r^2 < 0.5; F2 Pearson(r_R, d_T) >= 0.5 over
          kept patients; F3 r_R + d_T separates relapse, Mann-Whitney p < 0.01.

Data (gitignored, never committed): data/bruchovsky/dataTanaka/.

Run: JAX_PLATFORMS=cpu uv run python research/r2_bruchovsky_fit.py
     -> research/outputs/r2_bruchovsky_fit.{json,log}
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import mannwhitneyu, pearsonr

from adaptive_therapy.envs.lotka_volterra import vector_field
from adaptive_therapy.envs.sampling import STROBL_D_D, STROBL_R_S

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bruchovsky" / "dataTanaka"
OUT = Path(__file__).resolve().parent / "outputs" / "r2_bruchovsky_fit.json"
STEPS_PER_DAY = 10  # declared deviation (PREREG R2)
N_RESTARTS = 25
SEED = 0
LO = np.array([0.0, 0.0, 0.1, 1e-5])  # r_R/r_S, d_T/r_S, n0, f_R (Strobl p.6)
HI = np.array([1.0, 1.0, 1.0, 0.25])
NAMES = ("r_R", "d_T", "n0", "f_R")
R2_MIN = 0.5


def classes() -> dict[str, set[int]]:
    """Bruchovsky patient ids per class, from classificationsofpatients.txt."""
    text = (DATA / "classificationsofpatients.txt").read_text().split("Shaw_et_al")[0]
    out, key = {}, None
    for line in text.splitlines():
        s = line.strip().rstrip(":").lower()
        if s in ("without relapse", "with relapse") or s.startswith("with metastasis"):
            key = "metastasis" if "metastasis" in s else s
            out[key] = set()
        elif key and re.search(r"\d", s):
            out[key] |= {int(x) for x in re.findall(r"\d+", s)}
    return out


def load(path: Path) -> dict:
    """(days, psa, on) with days from the first row; psa NaN where unrecorded."""
    days, psa, on = [], [], []
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 9 or not row[8].strip():
                continue
            days.append(float(row[8]))
            psa.append(float(row[4]) if row[4].strip() else np.nan)
            on.append(float(row[7]) if row[7].strip() else np.nan)
    # 2026-09-30: patients 016 and 037 each carry one row out of date order (an
    # empty-PSA 1999/2001 row among 1995 rows); sort, never drop (PREREG R2 amendment).
    order = np.argsort(days, kind="stable")
    days = np.asarray(days)[order] - min(days)
    psa, on = np.asarray(psa)[order], np.asarray(on)[order]
    for i in range(len(on)):  # an unrecorded flag continues the previous one
        if np.isnan(on[i]):
            on[i] = on[i - 1] if i else 1.0
    return {"days": days, "psa": psa, "on": on}


def daily_schedule(p: dict) -> np.ndarray:
    """u on each day 0..last: the flag of the latest row at or before that day."""
    grid = np.arange(int(p["days"][-1]) + 1)
    idx = np.searchsorted(p["days"], grid, side="right") - 1
    return p["on"][idx]


@jax.jit
def trajectory(x, u_daily):
    """n(t)/n0 at every day, K = 1, fixed-step RK4. x = (r_R, d_T, n0, f_R)."""
    r_R, d_T, n0, f_R = x
    theta = jnp.stack(
        [STROBL_R_S, r_R * STROBL_R_S, 1.0, d_T * STROBL_R_S, d_T * STROBL_R_S, STROBL_D_D]
    )
    h = 1.0 / STEPS_PER_DAY
    y0 = jnp.stack([n0 * (1.0 - f_R), n0 * f_R])

    def day(y, u):
        def sub(z, _):
            f = lambda w: vector_field(0.0, w, (theta, u))  # noqa: E731
            k1 = f(z)
            k2 = f(z + h / 2 * k1)
            k3 = f(z + h / 2 * k2)
            k4 = f(z + h * k3)
            return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

        y2, _ = jax.lax.scan(sub, y, None, length=STEPS_PER_DAY)
        return y2, y.sum()

    _, n = jax.lax.scan(day, y0, u_daily)
    return n / n0


def fit(p: dict, pid: int) -> dict:
    u = jnp.asarray(daily_schedule(p))
    ok = ~np.isnan(p["psa"])
    at = jnp.asarray(p["days"][ok].astype(int))
    target = jnp.asarray(p["psa"][ok] / p["psa"][ok][0])

    res_fn = jax.jit(lambda x: trajectory(x, u)[at] - target)
    jac_fn = jax.jit(jax.jacfwd(res_fn))
    rng = np.random.default_rng([SEED, pid])
    runs = []
    for _ in range(N_RESTARTS):
        x0 = LO + rng.uniform(size=4) * (HI - LO)
        x0[3] = np.exp(rng.uniform(np.log(LO[3]), np.log(HI[3])))
        r = least_squares(
            lambda x: np.asarray(res_fn(jnp.asarray(x))),
            x0,
            jac=lambda x: np.asarray(jac_fn(jnp.asarray(x))),
            bounds=(LO, HI),
            method="trf",
            x_scale="jac",
        )
        runs.append((float(np.sum(r.fun**2)), r.x))
    runs.sort(key=lambda t: t[0])
    rss, best = runs[0]
    tgt = np.asarray(target)
    r2 = 1.0 - rss / max(float(np.sum((tgt - tgt.mean()) ** 2)), 1e-10)
    near = [x for s, x in runs if s <= rss * 1.01]
    return {
        "pid": pid,
        "n_obs": int(ok.sum()),
        "days": float(p["days"][-1]),
        **{k: float(v) for k, v in zip(NAMES, best, strict=True)},
        "rss": rss,
        "r2": float(r2),
        "rR_spread_near_best": float(max(x[0] for x in near) - min(x[0] for x in near)),
    }


def main() -> int:
    cl = classes()
    files = sorted((DATA / "Bruchovsky_et_al").glob("patient*.txt"))
    pids = [int(re.findall(r"\d+", f.stem)[0]) for f in files]
    fits = []
    for f, pid in zip(files, pids, strict=True):
        if pid in cl["metastasis"]:
            continue
        r = fit(load(f), pid)
        r["relapse"] = pid in cl["with relapse"]
        fits.append(r)
        print(f"  p{pid:03d} r2={r['r2']:.2f} r_R={r['r_R']:.2f} d_T={r['d_T']:.2f}", flush=True)

    kept = [r for r in fits if r["r2"] >= R2_MIN]
    rR, dT = np.array([r["r_R"] for r in kept]), np.array([r["d_T"] for r in kept])
    s = rR + dT
    rel = np.array([r["relapse"] for r in kept])
    mw = mannwhitneyu(s[rel], s[~rel], alternative="two-sided") if rel.any() else None
    n_exc = len(fits) - len(kept)
    rho = float(pearsonr(rR, dT)[0])
    f3_p = float(mw.pvalue) if mw else float("nan")
    in_box = (rR >= 0.5) & (dT <= 0.5)
    res = {
        "prereg": "adaptive_therapy/PREREG.md R2 [2026-09-30]",
        "n_patients": len(fits),
        "n_excluded": n_exc,
        "F1": 10 <= n_exc <= 20,
        "pearson_rR_dT": rho,
        "F2": rho >= 0.5,
        "mannwhitney_U": float(mw.statistic) if mw else None,
        "mannwhitney_p": f3_p,
        "n_relapse_kept": int(rel.sum()),
        "F3": f3_p < 0.01,
        "kill_fires": n_exc > 25,
        "descriptive": {
            "ranges": {
                k: [float(min(r[k] for r in kept)), float(max(r[k] for r in kept))] for k in NAMES
            },
            "median": {k: float(np.median([r[k] for r in kept])) for k in NAMES},
            "share_in_strobl_sweep": float(in_box.mean()),
            "share_indefinite_control": float(
                np.mean([r["d_T"] > (1 - 1.2 * r["n0"]) * r["r_R"] for r in kept])
            ),
            "share_rR_nonidentified": float(
                np.mean([r["rR_spread_near_best"] > 0.1 for r in kept])
            ),
            "median_r2_kept": float(np.median([r["r2"] for r in kept])),
        },
        "fits": fits,
    }
    res["pass"] = bool(res["F1"] and res["F2"] and res["F3"])
    print(
        f"F1 excluded {n_exc} (10-20): {res['F1']} | F2 Pearson {rho:.2f} (>=0.5): {res['F2']} | "
        f"F3 MW U={res['mannwhitney_U']} p={f3_p:.2g}: {res['F3']} -> PASS={res['pass']}"
    )
    print(json.dumps(res["descriptive"], indent=1))
    OUT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
