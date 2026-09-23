"""Phase 2 STOP-OR-GO gate: continuous-time vs discrete on LV, Tier-1 shift.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-22] + AMENDMENTs (Phase 2),
re-registered as Phase 2b [2026-09-23]: corrected trainer, closed-loop validity
gate (0), one-sided Wilcoxon; and Phase 2c [2026-09-23]: PQN trainer, capped
median. Select with the registration name; outputs are prefixed by it.

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

Run: JAX_PLATFORMS=cpu uv run python research/phase2_gate.py {phase2b | phase2c}

CPU, not GPU -- measured 2026-09-23 on an RTX 4090 vs this 32-core box, after
the PPO recompile fix: one update costs 5.5 s (MLP) / 6.4 s (ODE-RNN) on GPU
against 0.14 s / 0.37 s on CPU. Batch-32 rollouts of 52 sequential solves are
kernel-launch-bound, so the GPU is 17-40x slower (L166: 9x for bare rollouts).
"""

from __future__ import annotations

import importlib.util
import os

# MUST precede the jax import, so the backend is inferred from whether the CUDA
# plugin is installed (`make setup-gpu`) and not overridden by JAX_PLATFORMS=cpu.
ON_GPU = (
    importlib.util.find_spec("jax_cuda12_plugin") is not None
    and os.environ.get("JAX_PLATFORMS", "") != "cpu"
)

if ON_GPU:
    # 2026-09-23: GPU reductions (scatter-add in the PPO gradient) may reorder
    # float32 sums between runs. Forced deterministic so a shard is reproducible
    # and a resumed grid is not a mix of two numerical histories.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
else:
    # CPU only. 2026-09-22: added against "LLVM ERROR: Unable to allocate
    # section memory" with 21 GB free. L128 measured these caps to have NO
    # effect; the real cause was a per-update recompile in ppo.py exhausting
    # vm.max_map_count (fixed 2026-09-23). Kept, not deleted, so the CPU path
    # reproduces the configuration every Phase 2 attempt has run under.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "4")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import json
import subprocess
import sys
import time
from pathlib import Path

import equinox as eqx
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
from adaptive_therapy.rl import pqn, ppo
from adaptive_therapy.rl.rollout import EpisodeConfig, observation_dependence

OUT = Path(__file__).resolve().parent / "outputs"
CLOSED_LOOP_MIN = 0.10  # PREREG Phase 2b/2c criterion (0), fixed before either run
FAM = "lotka_volterra"
N_TRAIN = 200
N_TEST = 200
SEEDS = (0, 1, 2)
EXPOSURE_WEIGHTS = (0.05, 0.20)

# One entry per registration, so each stays reproducible from HEAD. Output files
# are prefixed by the key; Phase 2's run-2 artifacts remain phase2_*.
REGISTRATIONS = {
    "phase2b": {
        "prereg": "adaptive_therapy/PREREG.md Phase 2b [2026-09-23]",
        "trainer": lambda pol, th, y0, k, rw: ppo.train(pol, th, y0, k, ppo.PPOConfig(), EP, rw),
        "archs": {"mlp": MLPPolicy, "ode_rnn": ODERNNPolicy},
        "greedy_closed_loop": False,  # softmax policy: TV of action distributions
        "capped_median": False,  # registered estimator: median over progressors
    },
    "phase2c": {
        "prereg": "adaptive_therapy/PREREG.md Phase 2c [2026-09-23]",
        "trainer": lambda pol, th, y0, k, rw: pqn.train(pol, th, y0, k, pqn.PQNConfig(), EP, rw),
        # Q-networks with LayerNorm; MLP width 441 matches the ODE-RNN's Q path
        # (3090 vs 3091 params, critic heads unused by PQN).
        "archs": {
            "mlp": lambda k: MLPPolicy(k, width=441, layer_norm=True),
            "ode_rnn": lambda k: ODERNNPolicy(k, layer_norm=True),
        },
        "greedy_closed_loop": True,  # a Q-network acts by argmax
        "capped_median": True,  # min(TTP, horizon) over ALL patients
    },
}

EP = EpisodeConfig(n_visits=52, visit_interval=7.0, model_id=FAM)
TS = np.arange(EP.n_visits) * EP.visit_interval

# Tier-1 shift: theta OUTSIDE the training ranges. Fixed here, before any run.
TIER1 = {"r_S": 1.5, "r_R": 1.5, "K": 0.6}


def tier1_cohort(n: int, seed: int):
    """Sample, then push theta outside the training box.

    y0 is RE-DERIVED from the shifted K, keeping each patient's sampled
    resistant fraction f_R and fill level n_0 = (S+R)/K. 2026-09-23: run 1
    shifted K after y0 was built from the unshifted K, which put 86-99 of 200
    test patients ABOVE their carrying capacity. Those tumours shrink untreated
    (the control was 75% censored), and with d_D > 1 the drug term
    (1 - d_D u) * crowding turns positive above K, so full dose blew 28-35 of
    them up to NaN -- scored as censored. Scaling y0 by K'/K is exact because
    lotka_volterra.initial_state is linear in K.
    """
    c = sample_cohort(FAM, n, seed=seed + 1000)
    th = c.thetas.copy()
    names = list(THETA_RANGES[FAM].keys())
    for k, mult in TIER1.items():
        th[:, names.index(k)] *= mult
    k_idx = names.index("K")
    y0 = c.y0s * (th[:, k_idx] / c.thetas[:, k_idx])[:, None]
    return th, y0


def untreated_ttp(thetas, y0s):
    cfg = SolverConfig(t1=TS[-1], n_save=EP.n_visits)
    b = np.asarray(observe(solve_cohort(FAM, jnp.asarray(thetas), jnp.asarray(y0s), cfg=cfg), FAM))
    return ttp_running_nadir(b, TS)


def run_one(phase, arch_name, weight, seed):
    reg = REGISTRATIONS[phase]
    key = jax.random.PRNGKey(seed)
    k_init, k_train, k_eval = jax.random.split(key, 3)

    train_c = sample_cohort(FAM, N_TRAIN, seed=seed)
    policy = reg["archs"][arch_name](k_init)
    rw = RewardConfig(exposure_penalty=weight)

    t0 = time.perf_counter()
    policy, hist = reg["trainer"](
        policy, jnp.asarray(train_c.thetas), jnp.asarray(train_c.y0s), k_train, rw
    )
    train_s = time.perf_counter() - t0

    th, y0 = tier1_cohort(N_TEST, seed)
    ev = ppo.evaluate(policy, jnp.asarray(th), jnp.asarray(y0), k_eval, EP, rw)
    burden = np.asarray(ev["burden"])
    dose = np.asarray(ev["dose"])

    ttp = ttp_running_nadir(burden, TS)
    med, q1, q3 = ttp.median_iqr()
    cmed, cq1, cq3 = ttp.median_iqr_capped()
    pol_dir = OUT / f"{phase}_policies"
    pol_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(pol_dir / f"{arch_name}_w{weight}_s{seed}.eqx", policy)
    return {
        "observation_dependence": observation_dependence(
            policy, ev["obs"], EP, k_eval, greedy=reg["greedy_closed_loop"]
        ),
        "arch": arch_name,
        "backend": jax.default_backend(),
        "exposure_weight": weight,
        "seed": seed,
        "ttp_median": med,
        "ttp_iqr": [q1, q3],
        "ttp_median_capped": cmed,
        "ttp_iqr_capped": [cq1, cq3],
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


def _run_worker() -> int:
    """Execute ONE cell of the grid and write it to its own JSON shard.

    Each cell runs in a FRESH PROCESS, so no cell inherits another's JIT
    cache or memory mappings. Added 2026-09-22 against the LLVM section-memory
    crash; that crash turned out to be a per-update recompile in ppo.py that
    crossed vm.max_map_count WITHIN one cell (fixed 2026-09-23), so isolation
    alone never fixed it -- but it bounds any per-process growth, and is kept.
    """
    phase, arch, weight, seed = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
    shard = OUT / f"{phase}_shards" / f"{arch}_w{weight}_s{seed}.json"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(run_one(phase, arch, weight, seed)))
    return 0


def _spawn(phase, arch, weight, seed) -> dict:
    """Run one cell in a subprocess and read back its shard."""
    shard = OUT / f"{phase}_shards" / f"{arch}_w{weight}_s{seed}.json"
    if shard.exists():  # resume: a completed cell is never recomputed
        r = json.loads(shard.read_text())
        # ...unless it came from the other backend: CPU and GPU float32 histories
        # are not interchangeable (PREREG [2026-09-23] criterion iii).
        if r.get("backend") == jax.default_backend():
            return r

    # Retry in a fresh process. Added 2026-09-22 when L128 read the LLVM crash
    # as transient; it was deterministic (see _run_worker), so a retry of the
    # same cell fails the same way. Kept only as cheap insurance against a
    # genuinely transient infrastructure failure -- never against a bug.
    last = None
    for attempt in range(3):
        try:
            subprocess.run(
                [sys.executable, __file__, "--worker", phase, arch, str(weight), str(seed)],
                check=True,
                cwd=Path(__file__).resolve().parent.parent,
            )
            return json.loads(shard.read_text())
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            last = exc
            print(f"    retry {attempt + 1}/3 for {arch} w={weight} seed={seed}")
            time.sleep(20)
    raise RuntimeError(f"cell {arch} w={weight} seed={seed} failed 3x") from last


def main(phase: str) -> int:
    reg = REGISTRATIONS[phase]
    med_key = "ttp_median_capped" if reg["capped_median"] else "ttp_median"
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(f"{phase.upper()} STOP-OR-GO — LV, Tier-1 shift, exposure-matched")

    th0, y00 = tier1_cohort(N_TEST, 0)
    unt = untreated_ttp(th0, y00)
    unt_med = (unt.median_iqr_capped() if reg["capped_median"] else unt.median_iqr())[0]
    print(f"  untreated control (Tier-1): TTP {unt_med:.1f} d, censored {unt.censoring_rate:.0%}")

    runs = []
    for w in EXPOSURE_WEIGHTS:
        for arch in reg["archs"]:
            for s in SEEDS:
                r = _spawn(phase, arch, w, s)
                runs.append(r)
                print(
                    f"  w={w:<5.2f} {arch:8s} seed {s}: TTP {r[med_key]:6.1f} d "
                    f"exposure {r['exposure_median']:6.1f} "
                    f"dose {r['mean_dose']:.2f} closed-loop {r['observation_dependence']:.3f} "
                    f"({r['train_seconds']:.0f}s)"
                )

    # --- the gate ----------------------------------------------------------
    verdicts = {}
    for w in EXPOSURE_WEIGHTS:
        mlp = [r for r in runs if r["arch"] == "mlp" and r["exposure_weight"] == w]
        ode = [r for r in runs if r["arch"] == "ode_rnn" and r["exposure_weight"] == w]

        d_ttp = float(np.median([o[med_key] for o in ode]) - np.median([m[med_key] for m in mlp]))
        e_mlp = float(np.median([m["exposure_median"] for m in mlp]))
        e_ode = float(np.median([o["exposure_median"] for o in ode]))
        matched = abs(e_ode - e_mlp) / max(e_mlp, 1e-9) <= 0.10

        # Paired across patients, pooling seeds (same theta per seed index).
        a = np.concatenate([np.array(o["ttp_per_patient"]) for o in ode])
        b = np.concatenate([np.array(m["ttp_per_patient"]) for m in mlp])
        w_test = paired_wilcoxon(a, b, alternative="greater")  # ODE later than MLP

        beats_untreated = (
            np.median([o[med_key] for o in ode]) > unt_med
            and np.median([m[med_key] for m in mlp]) > unt_med
        )

        verdicts[str(w)] = {
            "delta_ttp_median": d_ttp,
            "exposure_mlp": e_mlp,
            "exposure_ode": e_ode,
            "exposure_matched_within_10pct": bool(matched),
            "wilcoxon": w_test,
            "both_beat_untreated": bool(beats_untreated),
            # Secondary, declared before run 2: the median is over PROGRESSORS
            # only, so it must be read beside how many patients progressed.
            "censoring_mlp": float(np.mean(np.concatenate([m["ttp_censored"] for m in mlp]))),
            "censoring_ode": float(np.mean(np.concatenate([o["ttp_censored"] for o in ode]))),
            # Phase 2c: >= on the CAPPED median -- a guard that the medians do
            # not contradict the paired test, which carries the direction; a tie
            # at the horizon (both arms >=50% censored) is "not worse".
            "criterion_1_ode_beats_mlp": bool(d_ttp >= 0 if reg["capped_median"] else d_ttp > 0),
            "criterion_2_significant": bool(w_test["p_value"] < 0.025),
            "pairs_ode_later": int((a > b).sum()),
            "pairs_ode_earlier": int((a < b).sum()),
        }

    # Criterion 1 is "ODE > MLP AT MATCHED EXPOSURE". Run 1's version of this
    # line omitted the match, so a TTP win bought with 3x the drug could have
    # printed PASS (2026-09-23; run 1 failed on TTP anyway).
    closed_loop = {
        f"{r['arch']}_w{r['exposure_weight']}_s{r['seed']}": r["observation_dependence"]
        for r in runs
    }
    valid = all(v >= CLOSED_LOOP_MIN for v in closed_loop.values())
    passed = valid and all(
        v["criterion_1_ode_beats_mlp"]
        and v["exposure_matched_within_10pct"]
        and v["criterion_2_significant"]
        and v["both_beat_untreated"]
        for v in verdicts.values()
    )
    if not valid:
        verdict = "INVALID — an arm is open-loop: training failure, no comparison"
    elif passed:
        verdict = "PASS — proceed to Phase 3"
    else:
        verdict = "FAIL — STOP, write the negative result"

    result = {
        "prereg": reg["prereg"],
        "ttp_estimator": med_key,
        "closed_loop_min": CLOSED_LOOP_MIN,
        "observation_dependence": closed_loop,
        "backend": jax.default_backend(),
        "verdict": verdict,
        "untreated_control_ttp": unt_med,
        "tier1_shift": TIER1,
        "per_weight": verdicts,
        "runs": runs,
    }
    (OUT / f"{phase}_gate.json").write_text(json.dumps(result, indent=2))

    print("-" * 74)
    for w, v in verdicts.items():
        print(
            f"  w={w}: dTTP {v['delta_ttp_median']:+7.1f} d  "
            f"exposure {v['exposure_ode']:.0f} vs {v['exposure_mlp']:.0f} "
            f"(matched={v['exposure_matched_within_10pct']})  "
            f"p(ODE later)={v['wilcoxon']['p_value']:.4f}  "
            f"pairs later/earlier {v['pairs_ode_later']}/{v['pairs_ode_earlier']}  "
            f"censored ode {v['censoring_ode']:.0%} mlp {v['censoring_mlp']:.0%}"
        )
    print(f"  VERDICT: {verdict}")
    print(
        f"  closed-loop (min {CLOSED_LOOP_MIN}): "
        + ", ".join(f"{k} {v:.2f}" for k, v in closed_loop.items())
    )
    print(f"  -> {OUT / f'{phase}_gate.json'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_run_worker())
    if len(sys.argv) != 2 or sys.argv[1] not in REGISTRATIONS:
        raise SystemExit(f"usage: phase2_gate.py {{{' | '.join(REGISTRATIONS)}}}")
    raise SystemExit(main(sys.argv[1]))
