"""Re-measure PROJECT.md §12's stiffness finding (L88) on GPU, paired against CPU.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-23].

L88 was measured ad hoc under CPU-XLA SIMD vectorisation -- the execution model
§12's warning is about -- so ROADMAP.md forbids carrying it over to GPU. Both
backends run here, in ONE process, on the SAME patients, so the comparison is
paired rather than across sessions.

  (i)   fixed-step dt0=0.05 vs 0.005 on flagged patients: max rel err <= 1e-3
  (ii)  Kvaerno5+PID at L88's tolerances fails on the flagged batch alone
  (iii) CPU<->GPU fixed-step burden parity <= 1e-4 max relative
  diag: Kvaerno5 at loose tolerance -- tests whether L88's failure is a float32
        precision floor (rtol 1e-6 is ~8x float32 eps) rather than the patients
  diag (AMENDMENT, post hoc): per-patient success of Kvaerno5 vs adaptive Tsit5
        over all kept patients, flagged vs bulk -- is the thrash a property of
        the fast-transient patients (L88's claim) or of the implicit solver?

Run: uv run --extra gpu python research/gpu_stiffness.py
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs.base import SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_cohort
from adaptive_therapy.envs.gate import bucket_by_stiffness, untreated_sanity_gate
from adaptive_therapy.envs.sampling import sample_cohort

OUT = Path(__file__).resolve().parent / "outputs"
FAM = "hahnfeldt"
N = 200
SEED = 0
CFG = SolverConfig(t0=0.0, t1=365.0, dt0=0.05, n_save=53)  # Phase 1's, unchanged
FINE = replace(CFG, dt0=0.005, max_steps=200_000)
TOLS = {"l88": (1e-6, 1e-8), "loose": (1e-4, 1e-6)}


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-10)))


def _burden(thetas, y0s, cfg: SolverConfig, device) -> np.ndarray:
    th = jax.device_put(jnp.asarray(thetas), device)
    y0 = jax.device_put(jnp.asarray(y0s), device)
    return np.asarray(observe(solve_cohort(FAM, th, y0, cfg=cfg), FAM))


def _adaptive(thetas, y0s, fine: np.ndarray, rtol: float, atol: float, device) -> dict:
    """Kvaerno5+PID on the flagged batch alone. Failure is a result, not an error."""
    # solve_one_patient hard-codes L88's tolerances for the stiff bucket, so the
    # loose arm needs its own controller; patched locally rather than widening
    # the package API for a one-shot diagnostic.
    import diffrax

    from adaptive_therapy.envs import cohort

    orig = diffrax.PIDController
    cohort.diffrax.PIDController = lambda **_: orig(rtol=rtol, atol=atol)
    t0 = time.perf_counter()
    try:
        b = _burden(thetas, y0s, replace(CFG, stiff=True, max_steps=200_000), device)
        ok = bool(np.isfinite(b).all())
        out = {"succeeded": ok, "error": None if ok else "non_finite"}
        if ok:
            out["max_rel_err_vs_fine_fixed"] = _rel(b, fine)
    except Exception as exc:  # diffrax raises on max_steps; that IS the finding
        msg = "max_steps" if "maximum number of solver steps" in str(exc) else str(exc)[-160:]
        out = {"succeeded": False, "error": f"{type(exc).__name__}: {msg}"}
    finally:
        cohort.diffrax.PIDController = orig
    out["seconds"] = time.perf_counter() - t0
    return out


def _per_patient(thetas, y0s, solver, rtol: float, atol: float, device) -> dict:
    """Which patients an adaptive solver finishes, one at a time (throw=False).

    `vmap` makes one failure fail the batch, which is why the arms above cannot
    say WHICH patients thrash. Written against diffrax directly because the
    package's solve path fixes the solver per bucket -- the thing under test.
    """
    import diffrax

    from adaptive_therapy.envs.cohort import family

    vf = family(FAM).vector_field

    def one(th, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf),
            solver,
            CFG.t0,
            CFG.t1,
            CFG.dt0,
            y0,
            args=(th, 0.0),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=200_000,
            throw=False,
        )
        ok = sol.result == diffrax.RESULTS.successful
        return ok, sol.stats["num_accepted_steps"], sol.stats["num_rejected_steps"]

    th = jax.device_put(jnp.asarray(thetas), device)
    y0 = jax.device_put(jnp.asarray(y0s), device)
    ok, acc, rej = (np.asarray(x) for x in jax.jit(jax.vmap(one))(th, y0))
    return {
        "n": int(ok.size),
        "n_succeeded": int(ok.sum()),
        "accepted_steps_median": float(np.median(acc)),
        "rejected_fraction_median": float(np.median(rej / np.maximum(acc + rej, 1))),
    }


def _time_fixed(thetas, y0s, device, reps: int = 5) -> float:
    _burden(thetas, y0s, CFG, device)  # compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _burden(thetas, y0s, CFG, device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise SystemExit("no GPU visible to JAX -- run with `uv run --extra gpu`")
    devices = {"cpu": jax.devices("cpu")[0], "gpu": gpus[0]}

    c = sample_cohort(FAM, N, seed=SEED)
    g = untreated_sanity_gate(FAM, c.thetas, c.y0s, CFG)
    th_f, y0_f = bucket_by_stiffness(c.thetas, c.y0s, g)["stiff"]
    th_k, y0_k = c.thetas[g.keep], c.y0s[g.keep]
    print(f"  {FAM}: kept {g.n_kept}/{N}, flagged fast-transient {len(th_f)}")

    report = {"prereg": "adaptive_therapy/PREREG.md [2026-09-23]", "n_flagged": len(th_f)}
    fixed = {}
    for name, dev in devices.items():
        coarse = _burden(th_f, y0_f, CFG, dev)
        fine = _burden(th_f, y0_f, FINE, dev)
        fixed[name] = _burden(th_k, y0_k, CFG, dev)
        r = {
            "fixed_coarse_vs_fine_max_rel": _rel(coarse, fine),
            "fixed_step_n200_seconds": _time_fixed(th_k, y0_k, dev),
            "adaptive": {k: _adaptive(th_f, y0_f, fine, *tol, dev) for k, tol in TOLS.items()},
        }
        report[name] = r
        print(
            f"  {name}: fixed coarse-vs-fine {r['fixed_coarse_vs_fine_max_rel']:.2e}  "
            f"fixed n={len(th_k)} {r['fixed_step_n200_seconds'] * 1e3:.1f} ms"
        )
        for k, a in r["adaptive"].items():
            print(f"        adaptive[{k}]: {a}")

    import diffrax

    th_b, y0_b = bucket_by_stiffness(c.thetas, c.y0s, g)["bulk"]
    diag = {}
    for sname, solver in (("kvaerno5", diffrax.Kvaerno5()), ("tsit5", diffrax.Tsit5())):
        for tname, tol in TOLS.items():
            for bname, (th, y0) in (("flagged", (th_f, y0_f)), ("bulk", (th_b, y0_b))):
                d = _per_patient(th, y0, solver, *tol, devices["cpu"])
                diag[f"{sname}_{tname}_{bname}"] = d
                print(f"  diag {sname:8s} {tname:5s} {bname:7s} {d}")
    report["diag_per_patient_cpu"] = diag

    report["cpu_gpu_parity_max_rel"] = _rel(fixed["gpu"], fixed["cpu"])
    gpu = report["gpu"]
    report["criteria"] = {
        "i_fixed_accurate_on_gpu": gpu["fixed_coarse_vs_fine_max_rel"] <= 1e-3,
        "ii_adaptive_l88_fails_on_gpu": not gpu["adaptive"]["l88"]["succeeded"],
        "iii_cpu_gpu_parity": report["cpu_gpu_parity_max_rel"] <= 1e-4,
    }
    report["transfers"] = all(report["criteria"].values())
    print(f"  parity CPU<->GPU {report['cpu_gpu_parity_max_rel']:.2e}")
    print(f"  criteria {report['criteria']}  -> transfers={report['transfers']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu_stiffness.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
