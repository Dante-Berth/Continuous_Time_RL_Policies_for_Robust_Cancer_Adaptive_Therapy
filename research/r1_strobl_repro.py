"""R1: reproduce Strobl 2021's published TTPs with our own LV vector field.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] R1, before this ran.
Verify, do not trust: `lotka_volterra.vector_field` (no PK, u = d) with Table 1's
parameters and the paper's CT / AT50 schedules must land on its numbers
(literature/papers/pmc-8455086.md: Fig. 4E p.24, text p.6-7).

Run: JAX_PLATFORMS=cpu uv run python research/r1_strobl_repro.py
     -> research/outputs/r1_strobl_repro.json
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs.lotka_volterra import vector_field
from adaptive_therapy.envs.sampling import STROBL_D_D, STROBL_R_S
from adaptive_therapy.eval.metrics import ttp_from_baseline

HORIZON_D = 3000
STEPS_PER_DAY = 20  # AT switching is checked every 0.05 d (declared risk)
OUT = Path(__file__).resolve().parent / "outputs"

FIG_4E = {  # Tumour 1, (n0, f_R) = (0.25, 0.001); None = never progresses
    "i": {"d_T": 0.0, "r_R": 1.0, "CT": 303, "AT": 323},
    "ii": {"d_T": 0.0, "r_R": 0.5, "CT": 575, "AT": 646},
    "iii": {"d_T": 0.5, "r_R": 1.0, "CT": 616, "AT": 848},
    "iv": {"d_T": 0.5, "r_R": 0.5, "CT": None, "AT": None},
}
GAINS = {  # (f_R, n0) -> AT - CT in days at d_T 0, no cost (p.6-7)
    (0.001, 0.25): 19,
    (0.001, 0.75): 211,
    (0.01, 0.25): 3,
    (0.01, 0.75): 104,
    (0.1, 0.25): 0,  # "indistinguishable": pass iff <= 3 d
    (0.1, 0.75): 0,
}


@partial(jax.jit, static_argnames="adaptive")
def simulate(n0, f_R, d_T, r_R, adaptive: bool) -> jnp.ndarray:
    """Total burden at days 1..HORIZON_D (K = 1, so burden is the density n)."""
    theta = jnp.array(
        [STROBL_R_S, r_R * STROBL_R_S, 1.0, d_T * STROBL_R_S, d_T * STROBL_R_S, STROBL_D_D]
    )
    h = 1.0 / STEPS_PER_DAY

    def rk4(y, d):
        f = lambda z: vector_field(0.0, z, (theta, d))  # noqa: E731
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def substep(carry, _):
        y, on = carry
        y = rk4(y, jnp.where(on, 1.0, 0.0))
        if adaptive:  # AT50 (p.5): off below 50% of n0, back on at n0
            n = y.sum()
            on = jnp.where(on & (n < 0.5 * n0), False, jnp.where(~on & (n >= n0), True, on))
        return (y, on), None

    def day(carry, _):
        carry, _ = jax.lax.scan(substep, carry, None, length=STEPS_PER_DAY)
        return carry, carry[0].sum()

    y0 = jnp.array([n0 * (1.0 - f_R), n0 * f_R])
    _, n = jax.lax.scan(day, (y0, jnp.array(True)), None, length=HORIZON_D)
    return n


def ttp(n0, f_R, d_T, r_R, adaptive: bool) -> float | None:
    """First day with n > 1.2 n0 (Strobl's RECIST rule); None if never."""
    n = np.concatenate([[n0], np.asarray(simulate(n0, f_R, d_T, r_R, adaptive))])
    res = ttp_from_baseline(n[None, :], np.arange(HORIZON_D + 1.0), confirm_days=0.0)
    return None if res.censored[0] else float(res.days[0])


def within(got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    return abs(got - want) <= max(3.0, 0.03 * want)


def main() -> int:
    rows = []
    for case, t in FIG_4E.items():
        for sched in ("CT", "AT"):
            got = ttp(0.25, 0.001, t["d_T"], t["r_R"], sched == "AT")
            rows.append({"target": f"Fig4E-{case}-{sched}", "want": t[sched], "got": got})
    for (f_R, n0), want in GAINS.items():
        ct, at = ttp(n0, f_R, 0.0, 1.0, False), ttp(n0, f_R, 0.0, 1.0, True)
        got = None if ct is None or at is None else at - ct
        ok_fn = (lambda g, w: g is not None and g <= 3.0) if f_R == 0.1 else within
        rows.append(
            {"target": f"gain-fR{f_R}-n0{n0}", "want": want, "got": got, "ok": ok_fn(got, want)}
        )
    for r in rows:
        r.setdefault("ok", within(r["got"], r["want"]))
        print(
            f"{r['target']:22s} want {str(r['want']):>5s}  got {str(r['got']):>7s}  {'ok' if r['ok'] else 'MISS'}"
        )
    passed = all(r["ok"] for r in rows)
    print(f"R1 {'PASS' if passed else 'FAIL'}: {sum(r['ok'] for r in rows)}/{len(rows)} targets")
    OUT.mkdir(exist_ok=True)
    (OUT / "r1_strobl_repro.json").write_text(
        json.dumps({"passed": passed, "rows": rows}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
