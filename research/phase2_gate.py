"""Phase 2 STOP-OR-GO gate: continuous-time vs discrete on LV, Tier-1 shift.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-22] + AMENDMENT.

  GATE (all must hold):
    1. median TTP(ODE) - median TTP(MLP) > 0 at MATCHED exposure (within 10%)
       on the Tier-1 shifted cohort
    2. paired Wilcoxon p < 0.05 across matched patients
    3. holds at >= 3 seeds and at BOTH exposure weights
    4. both arms beat the untreated control on TTP
  KILL: criterion 1 fails => STOP, write the negative result. No widening the
        network, no extra training, no third architecture.

  Declared grid: arch {MLP, ODE-RNN} x exposure weight {0.05, 0.20} x seed
  {0,1,2} = 12 runs. Width/LR fixed and NOT tuned. Params matched at 3076 each.

WHY EXPOSURE-MATCHED
Measured on LV: TTP rises monotonically with dose (u=0 -> 28 d, 0.25 -> 42,
0.5 -> 74, 0.75 -> 246, 1.0 -> 295). A pure-TTP gate is therefore won by
constant maximum dosing and would measure aggression, not scheduling skill.

Run: uv run python research/phase2_gate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs.base import SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_cohort
from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import THETA_RANGES, sample_cohort
from adaptive_therapy.eval.metrics import (
    burden_integral,
    cumulative_exposure,
    paired_wilcoxon,
    ttp_running_nadir,
)
from adaptive_therapy.policies.mlp import MLPPolicy
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy
from adaptive_therapy.rl.ppo import PPOConfig, evaluate, train
from adaptive_therapy.rl.rollout import EpisodeConfig

OUT = Path(__file__).resolve().parent / "outputs"
FAM = "lotka_volterra"
N_TRAIN = 200
N_TEST = 200
SEEDS = (0, 1, 2)
EXPOSURE_WEIGHTS = (0.05, 0.20)
ARCHS = {"mlp": MLPPolicy, "ode_rnn": ODERNNPolicy}

EP = EpisodeConfig(n_visits=52, visit_interval=7.0, model_id=FAM)
TS = np.arange(EP.n_visits) * EP.visit_interval

# Tier-1 shift: theta OUTSIDE the training ranges. Fixed here, before any run.
TIER1 = {"r_S": 1.5, "r_R": 1.5, "K": 0.6}


def tier1_cohort(n: int, seed: int):
    """Sample, then push theta outside the training box."""
    c = sample_cohort(FAM, n, seed=seed + 1000)
    th = c.thetas.copy()
    names = list(THETA_RANGES[FAM].keys())
    for k, mult in TIER1.items():
        th[:, names.index(k)] *= mult
    return th, c.y0s


def untreated_ttp(thetas, y0s):
    cfg = SolverConfig(t1=TS[-1], n_save=EP.n_visits)
    b = np.asarray(observe(solve_cohort(FAM, jnp.asarray(thetas), jnp.asarray(y0s), cfg=cfg), FAM))
    return ttp_running_nadir(b, TS)


def run_one(arch_name, weight, seed):
    key = jax.random.PRNGKey(seed)
    k_init, k_train, k_eval = jax.random.split(key, 3)

    train_c = sample_cohort(FAM, N_TRAIN, seed=seed)
    policy = ARCHS[arch_name](k_init)
    rw = RewardConfig(exposure_penalty=weight)

    t0 = time.perf_counter()
    policy, hist = train(
        policy,
        jnp.asarray(train_c.thetas),
        jnp.asarray(train_c.y0s),
        k_train,
        PPOConfig(),
        EP,
        rw,
    )
    train_s = time.perf_counter() - t0

    th, y0 = tier1_cohort(N_TEST, seed)
    ev = evaluate(policy, jnp.asarray(th), jnp.asarray(y0), k_eval, EP, rw)
    burden = np.asarray(ev["burden"])
    dose = np.asarray(ev["dose"])

    ttp = ttp_running_nadir(burden, TS)
    med, q1, q3 = ttp.median_iqr()
    return {
        "arch": arch_name,
        "exposure_weight": weight,
        "seed": seed,
        "ttp_median": med,
        "ttp_iqr": [q1, q3],
        "ttp_per_patient": ttp.days.tolist(),
        "ttp_censored": ttp.censored.tolist(),
        "censoring_rate": ttp.censoring_rate,
        "exposure_median": float(np.median(cumulative_exposure(dose, TS))),
        "exposure_per_patient": cumulative_exposure(dose, TS).tolist(),
        "burden_integral_median": float(np.median(burden_integral(burden, TS))),
        "mean_dose": float(dose.mean()),
        "train_seconds": train_s,
        "history": hist,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("PHASE 2 STOP-OR-GO — LV, Tier-1 shift, exposure-matched")

    th0, y00 = tier1_cohort(N_TEST, 0)
    unt = untreated_ttp(th0, y00)
    unt_med = unt.median_iqr()[0]
    print(f"  untreated control (Tier-1): TTP {unt_med:.1f} d, censored {unt.censoring_rate:.0%}")

    runs = []
    for w in EXPOSURE_WEIGHTS:
        for arch in ARCHS:
            for s in SEEDS:
                r = run_one(arch, w, s)
                runs.append(r)
                print(
                    f"  w={w:<5.2f} {arch:8s} seed {s}: TTP {r['ttp_median']:6.1f} d "
                    f"exposure {r['exposure_median']:6.1f} "
                    f"dose {r['mean_dose']:.2f} ({r['train_seconds']:.0f}s)"
                )

    # --- the gate ----------------------------------------------------------
    verdicts = {}
    for w in EXPOSURE_WEIGHTS:
        mlp = [r for r in runs if r["arch"] == "mlp" and r["exposure_weight"] == w]
        ode = [r for r in runs if r["arch"] == "ode_rnn" and r["exposure_weight"] == w]

        d_ttp = float(
            np.median([o["ttp_median"] for o in ode]) - np.median([m["ttp_median"] for m in mlp])
        )
        e_mlp = float(np.median([m["exposure_median"] for m in mlp]))
        e_ode = float(np.median([o["exposure_median"] for o in ode]))
        matched = abs(e_ode - e_mlp) / max(e_mlp, 1e-9) <= 0.10

        # Paired across patients, pooling seeds (same theta per seed index).
        a = np.concatenate([np.array(o["ttp_per_patient"]) for o in ode])
        b = np.concatenate([np.array(m["ttp_per_patient"]) for m in mlp])
        w_test = paired_wilcoxon(a, b)

        beats_untreated = (
            np.median([o["ttp_median"] for o in ode]) > unt_med
            and np.median([m["ttp_median"] for m in mlp]) > unt_med
        )

        verdicts[str(w)] = {
            "delta_ttp_median": d_ttp,
            "exposure_mlp": e_mlp,
            "exposure_ode": e_ode,
            "exposure_matched_within_10pct": bool(matched),
            "wilcoxon": w_test,
            "both_beat_untreated": bool(beats_untreated),
            "criterion_1_ode_beats_mlp": bool(d_ttp > 0),
            "criterion_2_significant": bool(w_test["p_value"] < 0.05),
        }

    passed = all(
        v["criterion_1_ode_beats_mlp"] and v["criterion_2_significant"] and v["both_beat_untreated"]
        for v in verdicts.values()
    )
    verdict = "PASS — proceed to Phase 3" if passed else "FAIL — STOP, write the negative result"

    result = {
        "prereg": "adaptive_therapy/PREREG.md [2026-09-22] + AMENDMENT",
        "verdict": verdict,
        "untreated_control_ttp": unt_med,
        "tier1_shift": TIER1,
        "per_weight": verdicts,
        "runs": runs,
    }
    (OUT / "phase2_gate.json").write_text(json.dumps(result, indent=2))

    print("-" * 74)
    for w, v in verdicts.items():
        print(
            f"  w={w}: dTTP {v['delta_ttp_median']:+7.1f} d  "
            f"exposure {v['exposure_ode']:.0f} vs {v['exposure_mlp']:.0f} "
            f"(matched={v['exposure_matched_within_10pct']})  "
            f"p={v['wilcoxon']['p_value']:.4f}"
        )
    print(f"  VERDICT: {verdict}")
    print(f"  -> {OUT / 'phase2_gate.json'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
