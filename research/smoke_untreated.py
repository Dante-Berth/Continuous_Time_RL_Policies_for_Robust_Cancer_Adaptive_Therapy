"""Untreated-trajectory smoke run across all three tumour ODE families.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-22].

  Success: all states finite; burden strictly positive; the three families are
           distinguishable by turning-point count or dynamic range.
  Kill:    any NaN, any negative burden, or two indistinguishable families.
  Declared search space: NONE. Nothing is tuned to make this pass.

WHY THIS SCRIPT EXISTS
Two jobs, and the second is the real one. It checks the simulator is sound, and
it exercises the /prereg -> run -> /logentry loop on something harmless before
that loop has to carry a result anyone cares about.

THE TRAP THIS DOES NOT TEST
Timing. Everything here is n=1 per family, so it says nothing about the
vmap-over-adaptive-solver cost in PROJECT.md 12. Do not read a duration from
this script and believe it generalises to a cohort.

Run: uv run python research/smoke_untreated.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from adaptive_therapy.envs import hahnfeldt, kuznetsov, lotka_volterra
from adaptive_therapy.envs.base import SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_one_patient

OUT = Path(__file__).resolve().parent / "outputs" / "smoke_untreated.json"

# Representative theta per family, from the literature values used in the tests.
PATIENTS = {
    "lotka_volterra": (
        jnp.array([0.027, 0.008, 1e4, 0.0, 0.0, 1.5]),
        lambda th: lotka_volterra.initial_state(th, f_R=0.01, n_0=0.5),
    ),
    "hahnfeldt": (
        jnp.array([0.192, 5.85, 0.00873, 0.66, 1e4]),
        lambda th: hahnfeldt.initial_state(th, v_0=100.0),
    ),
    "kuznetsov": (
        jnp.array([0.1181, 0.1245, 2.019e7, 3.422e-10, 0.00412, 0.018, 2.0e-9, 1.101e-7]),
        lambda th: kuznetsov.initial_state(th, t_0=1e5),
    ),
}

CFG = SolverConfig(t0=0.0, t1=365.0, dt0=0.05, n_save=366)


def summarise(model_id: str, burden: jnp.ndarray) -> dict:
    diffs = jnp.diff(burden)
    turning_points = int(jnp.sum(jnp.abs(jnp.diff(jnp.sign(diffs))) > 0))
    return {
        "model_id": model_id,
        "burden_t0": float(burden[0]),
        "burden_t1": float(burden[-1]),
        "burden_min": float(burden.min()),
        "burden_max": float(burden.max()),
        "dynamic_range_log10": float(
            jnp.log10(jnp.maximum(burden.max(), 1e-30))
            - jnp.log10(jnp.maximum(burden.min(), 1e-30))
        ),
        "turning_points": turning_points,
        "all_finite": bool(jnp.isfinite(burden).all()),
        "all_positive": bool((burden > 0).all()),
    }


def main() -> int:
    rows = []
    for model_id, (theta, init) in PATIENTS.items():
        y0 = init(theta)
        ys = solve_one_patient(model_id, theta, y0, dose=0.0, cfg=CFG)
        rows.append(summarise(model_id, observe(ys, model_id)))

    finite = all(r["all_finite"] for r in rows)
    positive = all(r["all_positive"] for r in rows)
    # Distinguishable: no two families share both a turning-point count and a
    # dynamic range within one order of magnitude.
    shapes = [(r["turning_points"], round(r["dynamic_range_log10"])) for r in rows]
    distinct = len(set(shapes)) == len(shapes)

    verdict = "PASS" if (finite and positive and distinct) else "FAIL"
    result = {
        "prereg": "adaptive_therapy/PREREG.md [2026-09-22]",
        "verdict": verdict,
        "criteria": {
            "all_finite": finite,
            "all_positive": positive,
            "families_distinguishable": distinct,
        },
        "shapes": {r["model_id"]: s for r, s in zip(rows, shapes)},
        "solver": CFG.to_dict(),
        "patients": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    print(f"SMOKE — untreated trajectories — {verdict}")
    for r in rows:
        print(
            f"  {r['model_id']:16s} burden {r['burden_t0']:.3e} -> {r['burden_t1']:.3e}"
            f"  range 1e{r['dynamic_range_log10']:.1f}  turns {r['turning_points']}"
        )
    print(f"  finite={finite}  positive={positive}  distinguishable={distinct}")
    print(f"  -> {OUT}")
    print("=" * 72)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
