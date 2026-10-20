"""W13 Stage 0 (PREREG W13, 2026-10-19; amended 2026-10-20): reproduce 7 qualitative
outcomes of de Pillis et al. 2006, Figs. 6-8, with `envs/depillis.py`. No policy.

Pass = 7/7. One reading only: no fitted rescaling, no second tolerance, no other
dose start day (CLOSED #4's rule). Outcomes are the operational definitions in
PREREG's IMPLEMENTATION NOTE as amended ("reaches capacity" -> T(120) >= 10 T0).

    uv run python research/w13_reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs import depillis as dp  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "w13_reproduce.json"
N_DAYS = 120


def pulses(days, strength, n=N_DAYS):
    v = np.zeros(n)
    v[list(days)] = strength
    return jnp.asarray(v)


def run(preset, y0, v_m=None, v_l=None):
    zero = jnp.zeros(N_DAYS)
    th = dp.theta_from(dp.PRESETS[preset])
    ys = dp.simulate_days(th, jnp.asarray(y0, float), zero if v_m is None else v_m,
                          zero if v_l is None else v_l, zero)  # fmt: skip
    return np.asarray(ys[:, 0])  # tumour T at the end of every day, index = day


def main() -> int:
    mouse = [1e6, 5e4, 100.0, 1.1e7, 0.0, 0.0]
    healthy = [1e6, 1e5, 1e2, 6e10, 0.0, 0.0]
    depleted = [1e6, 1e3, 10.0, 6e8, 0.0, 0.0]
    big = [2e7, 1e3, 10.0, 6e8, 0.0, 0.0]
    chemo14 = pulses(range(0, 98, 14), 1.0)  # 7 doses, 14-day cycle, 1 day each
    til = pulses([7], 8e8)  # 8e8 CD8+ cells from day 7 to 8
    inv_b = 1.0 / dp.MOUSE["b"]

    t_none = run("mouse", mouse)
    t_chemo = run("mouse", mouse, chemo14)
    t_comb = run("mouse", mouse, chemo14, til)
    t_healthy = run("patient9", healthy)
    t_depl = run("patient9", depleted)
    t_5d = run("patient9", big, pulses(range(0, 45, 5), 5.0))  # 9 doses, every 5 days
    t_10d = run("patient9", big, pulses(range(0, 90, 10), 5.0))  # 9 doses, every 10 days

    checks = {
        "fig6_no_treatment_reaches_plateau": (t_none[120] >= 10 * mouse[0], float(t_none[120])),
        "fig6_chemo_pulses_do_not_eradicate": (t_chemo[120] >= 1.0, float(t_chemo[120])),
        "fig6_chemo_plus_til_eradicates": (t_comb[120] < 1.0, float(t_comb[120])),
        "fig7_healthy_patient9_eradicates": (t_healthy[120] < 1.0, float(t_healthy[120])),
        "fig7_depleted_patient9_fails_to_kill": (t_depl[120] >= 10 * depleted[0], float(t_depl[120])),
        "fig8_chemo_every_5d_eradicates_by_day_55": (bool((t_5d[:56] < 1.0).any()), float(t_5d[:56].min())),
        "fig8_chemo_every_10d_regrows": (bool(t_10d[120] >= 1.0 and t_10d[120] > t_10d[119]), float(t_10d[120])),
    }  # fmt: skip
    res = {
        "prereg": "W13 Stage 0, 2026-10-19, amended 2026-10-20",
        "inv_b_mouse": float(inv_b),
        "checks": {k: {"pass": bool(v[0]), "value": v[1]} for k, v in checks.items()},
    }
    res["n_pass"] = sum(c["pass"] for c in res["checks"].values())
    res["stage0_pass"] = res["n_pass"] == len(checks)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for k, c in res["checks"].items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {k:48s} {c['value']:.3e}")
    print(f"{res['n_pass']}/{len(checks)}  stage0_pass={res['stage0_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
