"""V5: do our simulators reproduce Hahnfeldt 1999 and Zhang 2017 from their pages?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-02] V5 (cd55e4d), before
any V5 simulation ran. Rules and fixed schedules only, no policy.

  V5a  envs/hahnfeldt.vector_field, unchanged, driven with u = 1 and e -> e*g(t),
       g the paper's bolus PK (Cancer Res 59:4770, Eq. C3, p.4773). Targets
       H1-H5 (p.4771-4774).
  V5c  envs/zhang2017.vector_field (Nat Commun 8:1816, Eq. 1-2, p.7-8).
       Patient #2: 5% T- at 323, 90% T- at 784 under MTD, never under adaptive
       (Table 1, p.7).

Run: JAX_PLATFORMS=cpu uv run python research/v5_repro.py
     -> research/outputs/v5_repro.{json,log}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs import hahnfeldt, zhang2017

jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).resolve().parent / "outputs"
RESULT = OUT / "v5_repro.json"
SOLVER = diffrax.Tsit5()
CTRL = diffrax.PIDController(rtol=1e-8, atol=1e-8)

# --- V5a: Hahnfeldt ----------------------------------------------------------

# Table 1, p.4771. theta = (a = lambda1, b, d, e, K_max); e is set per interval.
LAMBDA1, B, D_INH, K0 = 0.192, 5.85, 0.00873, 625.0
AGENTS = {"tnp470": (1.3, 10.1), "endostatin": (0.66, 1.7), "angiostatin": (0.15, 0.38)}
N_SUB = 10  # V samples per dosing interval for the set-point mean


def hf_interval(y, g0, e, clr, length):
    """Integrate one dosing interval with g(t) = g0 exp(-clr t) (no new dose)."""

    def f(t, yy, args):
        theta = jnp.array([LAMBDA1, B, D_INH, e * g0 * jnp.exp(-clr * t), 0.0])
        return hahnfeldt.vector_field(t, yy, (theta, 1.0))

    ts = jnp.linspace(0.0, length, N_SUB + 1)[1:]
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), SOLVER, 0.0, length, 1e-3, y,
        saveat=diffrax.SaveAt(ts=ts), stepsize_controller=CTRL, max_steps=100_000,
    )  # fmt: skip
    return sol.ys


def hf_run(v0, k0, agent, dose, every, days):
    """Boli of `dose` mg/kg every `every` days from day 0; returns (t, V) samples."""
    e, clr = AGENTS[agent] if agent else (0.0, 1.0)
    n = int(round(days / every))
    step = jax.jit(lambda y, g: hf_interval(y, g, e, clr, every))
    y, g, ts, vs = jnp.array([v0, k0]), 0.0, [], []
    for i in range(n):
        g = g + dose
        ys = np.asarray(step(y, g))
        ts += list(i * every + np.linspace(0, every, N_SUB + 1)[1:])
        vs += list(ys[:, 0])
        y, g = jnp.asarray(ys[-1]), g * np.exp(-clr * every)
    return np.array(ts), np.array(vs)


def hf_continuous(v0, k0, agent, rate, days):
    """Constant infusion `rate` mg/kg/day: g = (rate/clr)(1 - exp(-clr t))."""
    e, clr = AGENTS[agent]

    def f(t, yy, args):
        g = rate / clr * (1.0 - jnp.exp(-clr * t))
        theta = jnp.array([LAMBDA1, B, D_INH, e * g, 0.0])
        return hahnfeldt.vector_field(t, yy, (theta, 1.0))

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), SOLVER, 0.0, days, 1e-3, jnp.array([v0, k0]),
        stepsize_controller=CTRL, max_steps=100_000,
    )  # fmt: skip
    return float(sol.ys[-1, 0])


def set_point(ts, vs, last=100.0):
    return float(vs[ts > ts[-1] - last].mean())


def k0_for(v0):
    """K0 = 625 for the experiments' small starts (Table 1); K0 = V0 for the
    large starts of Fig. 4 II C-D, whose K0 the paper does not give. Set points
    are claimed independent of start (p.4774), so this choice is tested, not
    assumed."""
    return K0 if v0 <= K0 else v0


def within(got, want, tol):
    return abs(got - want) <= tol * want


def v5a() -> dict:
    res = {}
    t, v = hf_run(200.0, K0, None, 0.0, 1.0, 2000)
    res["H1"] = {"got": set_point(t, v), "want": 17400.0, "analytic": (B / D_INH) ** 1.5}
    res["H1"]["pass"] = within(res["H1"]["got"], 17400.0, 0.03)
    sp = {
        v0: set_point(*hf_run(v0, k0_for(v0), "tnp470", 30.0, 2.0, 2000))
        for v0 in (170.0, 12300.0, 17300.0)
    }
    res["H2"] = {
        "got": sp,
        "want": 12300.0,
        "pass": all(within(x, 12300.0, 0.03) for x in sp.values()),
    }
    t, v = hf_run(177.0, K0, "angiostatin", 14.0, 1.0, 2000)
    res["H3"] = {"got": set_point(t, v), "want": 240.0, "overshoot_max": float(v.max())}
    res["H3"]["pass"] = within(res["H3"]["got"], 240.0, 0.03) and v.max() > 177.0
    t, v = hf_run(200.0, K0, "tnp470", 30.0, 2.0, 14)
    bolus13 = float(np.interp(13.0, t, v))
    cont13 = hf_continuous(200.0, K0, "tnp470", 15.0, 13.0)
    res["H4"] = {"got": [bolus13, cont13], "want": [1840.0, 1300.0]}
    res["H4"]["pass"] = within(bolus13, 1840.0, 0.10) and within(cont13, 1300.0, 0.10)
    d15 = {}
    for name, agent, dose, every in (
        ("endostatin20", "endostatin", 20.0, 1.0),
        ("endostatin4", "endostatin", 4.0, 1.0),
        ("tnp470_30qod", "tnp470", 30.0, 2.0),
    ):
        t, v = hf_run(200.0, K0, agent, dose, every, 16)
        d15[name] = float(np.interp(15.0, t, v))
    res["H5"] = {
        "got": d15,
        "pass": d15["endostatin20"] < 200.0
        and d15["endostatin4"] > 200.0
        and d15["tnp470_30qod"] > 200.0,
    }
    return res


# --- V5c: Zhang --------------------------------------------------------------

DAYS = 10_000


def zh_step(theta):
    @jax.jit
    def step(y, drug):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(zhang2017.vector_field), SOLVER, 0.0, 1.0, 0.1, y,
            args=(theta, drug), stepsize_controller=CTRL, max_steps=100_000,
        )  # fmt: skip
        return sol.ys[-1]

    return step


def zh_equilibrium(theta) -> np.ndarray:
    """Untreated equilibrium, by integration from an even mix (100,000 days)."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(zhang2017.vector_field), SOLVER, 0.0, 100_000.0, 0.1,
        jnp.array([1000.0, 1000.0, 1000.0, 0.0]), args=(theta, 0.0),
        stepsize_controller=CTRL, max_steps=1_000_000,
    )  # fmt: skip
    return np.asarray(sol.ys[-1])


def zh_run(theta, regimen: str) -> dict:
    eq = zh_equilibrium(theta)
    psa_eq = eq[3]
    y = jnp.asarray(np.r_[0.25 * eq[:3], 0.25 * psa_eq])
    step = zh_step(theta)
    start, base, drug = None, None, 0.0
    frac, psa, drugs = [], [], []
    for day in range(DAYS):
        x = np.asarray(y)
        if start is None and x[3] >= 0.8 * psa_eq:
            start, base, drug = day, x[3], 1.0
        elif start is not None and regimen == "adaptive":
            drug = 0.0 if x[3] < 0.5 * base else 1.0 if x[3] >= base else drug
        frac.append(x[2] / max(x[:3].sum(), 1e-10))
        psa.append(x[3])
        drugs.append(drug)
        y = step(y, drug)
    frac = np.array(frac)

    def first(th):
        idx = np.flatnonzero(frac >= th)
        return int(idx[0]) if idx.size else None

    return {
        "equilibrium": eq.tolist(),
        "start": start,
        "t_5pct": first(0.05),
        "t_90pct": first(0.90),
        "drug_share": float(np.mean(drugs[start:])) if start is not None else None,
    }


def v5c() -> dict:
    res = {}
    for name, alphas in (("patient1", zhang2017.PATIENT_1), ("patient2", zhang2017.PATIENT_2)):
        theta = zhang2017.theta_for(alphas)
        res[name] = {reg: zh_run(theta, reg) for reg in ("MTD", "adaptive")}
    p2 = res["patient2"]

    def near(got, want):
        return got is not None and abs(got - want) <= max(0.03 * want, 3)

    res["targets_patient2"] = {
        "t5_MTD": [p2["MTD"]["t_5pct"], 323, near(p2["MTD"]["t_5pct"], 323)],
        "t5_adaptive": [p2["adaptive"]["t_5pct"], 323, near(p2["adaptive"]["t_5pct"], 323)],
        "t90_MTD": [p2["MTD"]["t_90pct"], 784, near(p2["MTD"]["t_90pct"], 784)],
        "t90_adaptive_never": [p2["adaptive"]["t_90pct"], None, p2["adaptive"]["t_90pct"] is None],
    }
    res["pass"] = all(v[2] for v in res["targets_patient2"].values())
    eq1 = res["patient1"]["MTD"]["equilibrium"]
    res["patient1_T-_at_equilibrium"] = eq1[2]
    return res


def main() -> int:
    res = {"prereg": "adaptive_therapy/PREREG.md V5 [2026-10-02]", "V5a": v5a(), "V5c": v5c()}
    a = res["V5a"]
    print("== V5a Hahnfeldt 1999 ==")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        print(
            f"  {h}: got {a[h]['got']}  want {a[h].get('want', 'qualitative')}  PASS={a[h]['pass']}"
        )
    print(
        f"  H1 analytic (b/d)^1.5 = {a['H1']['analytic']:.0f}; H3 overshoot max {a['H3']['overshoot_max']:.0f}"
    )
    res["V5a"]["pass"] = all(a[h]["pass"] for h in ("H1", "H2", "H3", "H4", "H5"))
    print(f"  V5a reproduces: {res['V5a']['pass']}")
    c = res["V5c"]
    print("== V5c Zhang 2017 ==")
    for p in ("patient1", "patient2"):
        for reg, r in c[p].items():
            print(
                f"  {p} {reg:8s} start {r['start']}  5% T- {r['t_5pct']}  90% T- {r['t_90pct']}"
                f"  drug share {r['drug_share']:.3f}  eq {np.round(r['equilibrium'], 1).tolist()}"
            )
    for k, v in c["targets_patient2"].items():
        print(f"  {k:20s} got {v[0]}  want {v[1]}  ok={v[2]}")
    print(f"  patient 1 T- at equilibrium {c['patient1_T-_at_equilibrium']:.3g}")
    print(f"  V5c reproduces: {c['pass']}")
    RESULT.write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
