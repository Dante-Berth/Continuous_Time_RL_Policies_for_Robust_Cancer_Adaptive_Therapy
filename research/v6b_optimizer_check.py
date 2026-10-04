"""V6b (post hoc, NOT pre-registered): is V6-gate's failure the optimiser or the model?

Re-starts least_squares from each V6 best fit with 10x the evaluation budget
(3000) and tight tolerances. A large objective drop means V6's fits were
optimiser-limited, so the gate's kill would be on our protocol, not on the model.
Diagnosis only: V6-gate's verdict is not re-graded by this script.

Run: JAX_PLATFORMS=cpu uv run python research/v6b_optimizer_check.py
     -> research/outputs/v6b_optimizer_check.json
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import research.v6_phan_fit as v

PIDS = (28, 29, 77, 17, 25, 101)  # Phan's examples that pass the rule + two more
OUT = v.OUT.with_name("v6b_optimizer_check.json")


def refit(pid: int) -> dict:
    prev = {r["pid"]: r for r in json.loads(v.OUT.read_text())["fits"]}[pid]
    p = v.load(v.DATA / f"patient{pid:03d}.txt")
    fit_m = p["phase"] <= 2
    u = jnp.asarray(v.daily_dose(p))
    at, obs, w = v._padded(p, fit_m, 128)
    first = lambda c: p[c][np.isfinite(p[c])][0]  # noqa: E731
    y = jnp.array([first("tst"), first("psa")])
    z0 = np.clip(np.log([prev["params"][k] for k in v.NAMES]), v.LO + 1e-9, v.HI - 1e-9)
    args = (y, u, at, obs, w)
    r = least_squares(
        lambda z: np.asarray(v.RES(jnp.asarray(z), *args)), z0,
        jac=lambda z: np.asarray(v.JAC(jnp.asarray(z), *args)),
        bounds=(v.LO, v.HI), method="trf", x_scale="jac", max_nfev=3000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )  # fmt: skip
    ys = np.asarray(v.SIM(jnp.asarray(r.x), y, u))
    f_psa, f_and = v.rmse(ys, p, fit_m)
    at_bound = int(np.sum((r.x - v.LO < 1e-3) | (v.HI - r.x < 1e-3)))
    return {
        "pid": pid, "obj_v6": prev["objective"], "obj_refit": float(np.sum(r.fun**2)),
        "nfev": int(r.nfev), "status": int(r.status),
        "fit_psa_v6": prev["fit_rmse_psa"], "fit_psa_refit": f_psa,
        "fit_and_v6": prev["fit_rmse_and"], "fit_and_refit": f_and,
        "n_params_at_bound": at_bound,
        "at_bound": [k for k, z in zip(v.NAMES, r.x, strict=True)
                     if z - v.LO[v.NAMES.index(k)] < 1e-3 or v.HI[v.NAMES.index(k)] - z < 1e-3],
    }  # fmt: skip


def main() -> int:
    with Pool(len(PIDS)) as pool:
        res = pool.map(refit, PIDS)
    OUT.write_text(json.dumps(res, indent=1))
    for r in res:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
