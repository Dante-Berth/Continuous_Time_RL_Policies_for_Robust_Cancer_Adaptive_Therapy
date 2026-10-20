"""W13 Stage 1 (PREREG W13, 2026-10-19): rule-based headroom on the de Pillis 2006
family. No policy is trained. Run only after Stage 0 passes (it did, as amended;
RESEARCH_LOG 2026-10-20).

Chemo only (v_L = v_I = 0). Visits every 5 d, 24 visits (120 d); dose a in the rule's
set times the paper's pulse (v_M = a for 1 day, then 4 days off); observation = log T
with 5% log-normal noise, T only. Score U = min(escape day, 120) - w * sum(a), escape =
first day T >= 0.1/b; T < 1 cell counts as no escape. Rules: none, MTD, constant
1/4, 1/2, 3/4, and thr_x (full dose when the observation exceeds x times the first
one) for x in {0.3, 0.5, 0.8, 1.2}. "Per-patient best" = best rule on that patient's own
scored rollout under the SAME noise draw (O1's optimistic convention).

Gate at w = 0.1, seed+patient bootstrap, both must have lower CI > +10 d:
  P1 per-patient best - MTD;  P2 per-patient best - the single best rule chosen on the
  seed's training cohort (200 patients).

    uv run python research/w13_headroom.py [--smoke]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import w4_irregular as w4  # noqa: E402

from adaptive_therapy.envs import depillis as dp  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "w13_headroom.json"
SEEDS = (1, 2, 3)
N_PAT = 200
WEIGHTS = (0.05, 0.1, 0.2, 0.4)
GATE_W = 0.1
N_VISITS, VISIT_DAYS, HORIZON = 24, 5, 120
NOISE = 0.05
FIXED_DT = 0.05
GATE_MARGIN = 10.0
# (name, dose when below the threshold, dose when above, threshold x first observation)
RULES = (
    ("none", 0.0, 0.0, np.inf),
    ("MTD", 1.0, 1.0, 0.0),
    ("c0.25", 0.25, 0.25, 0.0),
    ("c0.5", 0.5, 0.5, 0.0),
    ("c0.75", 0.75, 0.75, 0.0),
    ("thr0.3", 0.0, 1.0, 0.3),
    ("thr0.5", 0.0, 1.0, 0.5),
    ("thr0.8", 0.0, 1.0, 0.8),
    ("thr1.2", 0.0, 1.0, 1.2),
)
RULE_ARR = jnp.array([[lo, hi, x] for _, lo, hi, x in RULES])


def episode(theta, y0, rule, key):
    """One patient, one rule. Returns (escape day, cumulative dose, final T, daily T)."""
    lo, hi, x = rule
    eps = NOISE * jax.random.normal(key, (N_VISITS,))
    zero4 = jnp.zeros(VISIT_DAYS - 1)

    def visit(carry, e):
        y, obs0 = carry
        obs = y[0] * jnp.exp(e)
        obs0 = jnp.where(obs0 < 0, obs, obs0)  # first observation is the reference
        a = jnp.where(obs > x * obs0, hi, lo)
        v_m = jnp.concatenate([a[None], zero4])
        zeros = jnp.zeros(VISIT_DAYS)
        ys = dp.simulate_days(theta, y, v_m, zeros, zeros, fixed_dt=FIXED_DT)
        return (ys[-1], obs0), (ys[1:, 0], a)

    (_, _), (t_days, doses) = jax.lax.scan(visit, (y0, jnp.asarray(-1.0)), eps)
    t_days = t_days.reshape(-1)  # daily T, days 1..HORIZON
    over = t_days >= 0.1 / theta[1]
    escape = jnp.where(over.any(), jnp.argmax(over) + 1, HORIZON).astype(float)
    return escape, doses.sum(), t_days[-1], t_days


@jax.jit
def run_cohort(theta, y0, keys):
    """(rules, patients) arrays of escape day and exposure, plus eradicated flag."""
    per_rule = jax.vmap(episode, in_axes=(None, None, 0, None))
    per_pat = jax.vmap(per_rule, in_axes=(0, 0, None, 0))
    esc, expo, t_end, _ = per_pat(theta, y0, RULE_ARR, keys)
    return esc.T, expo.T, (t_end < 1.0).T  # (rules, patients)


def utilities(esc, expo):
    """U[rule, w, patient]."""
    return (
        np.minimum(esc, HORIZON)[:, None, :] - np.array(WEIGHTS)[None, :, None] * expo[:, None, :]
    )


def cohort(seed: int, which: int, n: int):
    key = jax.random.fold_in(jax.random.PRNGKey(seed), which)  # 0 = train, 1 = test
    theta, y0 = dp.sample_patients(key, n)
    keys = jax.random.split(jax.random.fold_in(key, 7), n)
    esc, expo, erad = run_cohort(theta, y0, keys)
    return np.asarray(esc), np.asarray(expo), np.asarray(erad)


def summarise(x):
    q = np.percentile(x, [25, 50, 75])
    return {"median": float(q[1]), "iqr": [float(q[0]), float(q[2])]}


def main(smoke: bool = False) -> int:
    n = 8 if smoke else N_PAT
    wi = WEIGHTS.index(GATE_W)
    names = [r[0] for r in RULES]
    p1, p2, per_seed = [], [], []
    detail = {}
    for seed in (99,) if smoke else SEEDS:  # smoke: unregistered seed 99
        esc_tr, expo_tr, _ = cohort(seed, 0, n)
        esc_te, expo_te, erad_te = cohort(seed, 1, n)
        u_tr, u_te = utilities(esc_tr, expo_tr), utilities(esc_te, expo_te)
        single = int(np.argmax(u_tr[:, wi].mean(axis=1)))  # chosen on training patients only
        best_pp = u_te[:, wi].max(axis=0)
        p1.append(best_pp - u_te[names.index("MTD"), wi])
        p2.append(best_pp - u_te[single, wi])
        per_seed.append(
            {
                "seed": seed,
                "single_rule": names[single],
                "p1": float(p1[-1].mean()),
                "p2": float(p2[-1].mean()),
            }
        )
        detail[seed] = {
            "mean_U_test_by_rule_w0.1": {k: float(u_te[i, wi].mean()) for i, k in enumerate(names)},
            "eradicated_share_by_rule": {k: float(erad_te[i].mean()) for i, k in enumerate(names)},
            "best_rule_share": {k: float((u_te[:, wi].argmax(axis=0) == i).mean()) for i, k in enumerate(names)},
            "U_w0.1_MTD": summarise(u_te[names.index("MTD"), wi]),
            "U_w0.1_best_per_patient": summarise(best_pp),
            "exposure_MTD": summarise(expo_te[names.index("MTD")]),
            "escape_MTD": summarise(esc_te[names.index("MTD")]),
        }  # fmt: skip
        print(per_seed[-1], flush=True)
        if smoke:
            print(json.dumps(detail[seed], indent=1))
    res = {
        "prereg": "W13 Stage 1, 2026-10-19",
        "smoke": smoke,
        "per_seed": per_seed,
        "detail": detail,
    }
    if not smoke:
        for key, arr in (("P1_best_minus_MTD", p1), ("P2_best_minus_single", p2)):
            d = np.stack(arr)[None, None]  # (w=1, p=1, seed, patient)
            both, patients = w4.delta_draws(d)
            lo, hi = np.percentile(both, [2.5, 97.5])
            res[key] = {"delta": float(np.mean(arr)), "ci95": [float(lo), float(hi)], "ci95_patients": np.percentile(patients, [2.5, 97.5]).tolist(), "pass": bool(lo > GATE_MARGIN)}  # fmt: skip
        res["gate_pass"] = bool(
            res["P1_best_minus_MTD"]["pass"] and res["P2_best_minus_single"]["pass"]
        )
        print(
            json.dumps(
                {k: res[k] for k in ("P1_best_minus_MTD", "P2_best_minus_single", "gate_pass")},
                indent=1,
            )
        )
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--smoke" in sys.argv))
