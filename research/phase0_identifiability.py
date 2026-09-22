"""Phase 0 — practical identifiability of theta, and separability of the three
families, from burden-only observation.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-22]. Criteria, verbatim:

  (1) Identifiability: Fisher eigenspectrum + per-component profile-likelihood
      classification into identifiable / weak / flat. NO pass-fail threshold --
      it is a measurement.
  (2) Separability (THE GATE): nearest-neighbour classifier on log-burden
      trajectories alone, balanced accuracy >= 0.80 on a held-out third.
  (3) Trajectory collision: fraction of cross-family pairs agreeing within
      relative L2 < 5%, reported PER PAIR.

  KILL: balanced accuracy < 0.80 => families are not separable under
  burden-only; STOP and reframe Q3 before writing any encoder. Do not rescue by
  lengthening the window, widening ranges, or adding an observable.

  Declared window: t in [0, 365] d, 52 weekly samples. MAY NOT be changed after
  seeing the accuracy -- see the registration for why that would be a selection
  event. Seeds 0,1,2; report median + range.

WHY A NEAREST-NEIGHBOUR CLASSIFIER AND NOT SOMETHING STRONGER
The question is whether the burden signal CONTAINS the family information, not
whether a large model can extract it. A 1-NN on normalised log-burden is close
to a lower bound on extractability: if 1-NN separates them, the information is
there in an almost trivially accessible form. A deep classifier scoring well
would leave "the information is there" and "we built something clever"
confounded, which is the wrong trade for a gate.

Run: uv run python research/phase0_identifiability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs.base import SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_cohort
from adaptive_therapy.envs.sampling import THETA_RANGES, sample_cohort

OUT = Path(__file__).resolve().parent / "outputs"
FAMILIES = ("lotka_volterra", "hahnfeldt", "kuznetsov")

# Declared in the registration. Do not edit to improve a number.
T1 = 365.0
N_SAMPLES = 52
NOISE_SIGMA = 0.05
N_PER_FAMILY = 300
SEEDS = (0, 1, 2)
ACCURACY_GATE = 0.80
COLLISION_TOL = 0.05

CFG = SolverConfig(t0=0.0, t1=T1, dt0=0.05, n_save=N_SAMPLES)


# --- observation model ------------------------------------------------------


def burdens(model_id: str, thetas: np.ndarray, y0s: np.ndarray) -> np.ndarray:
    """Noise-free burden trajectories, (n, N_SAMPLES)."""
    ys = solve_cohort(model_id, jnp.asarray(thetas), jnp.asarray(y0s), cfg=CFG)
    return np.asarray(observe(ys, model_id))


def add_noise(b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative lognormal noise: measurement error is relative."""
    return b * np.exp(rng.normal(0.0, NOISE_SIGMA, size=b.shape))


def featurise(b: np.ndarray) -> np.ndarray:
    """Normalised log-burden shape.

    Log because burden spans decades; subtracting the first sample so the
    classifier keys on SHAPE rather than on absolute scale -- scale is a
    property of the initial condition, not of the mechanism, and letting the
    classifier use it would inflate the accuracy for the wrong reason.
    """
    lb = np.log(np.maximum(b, 1e-30))
    lb = lb - lb[:, :1]
    scale = np.maximum(np.abs(lb).max(axis=1, keepdims=True), 1e-12)
    return lb / scale


# --- (1) identifiability ----------------------------------------------------


def fisher_and_profiles(model_id: str, theta: np.ndarray, y0: np.ndarray) -> dict:
    """Fisher eigenspectrum + per-component profile-likelihood classification.

    Sensitivities by forward-mode AD through the solver; under multiplicative
    lognormal noise the natural residual is in log space, so the FIM is built
    from d(log burden)/d(log theta) -- which also makes components with
    different units comparable.
    """

    def log_burden(log_theta):
        th = jnp.exp(log_theta)
        ys = solve_cohort(model_id, th[None], jnp.asarray(y0)[None], cfg=CFG)
        return jnp.log(jnp.maximum(observe(ys, model_id)[0], 1e-30))

    lt = jnp.log(jnp.maximum(jnp.asarray(theta), 1e-30))
    # jacREV, not jacfwd: diffrax's default adjoint is a custom_vjp, so
    # forward-mode cannot differentiate through it. Reverse-mode is also the
    # right choice on shape -- 52 outputs vs 5-8 parameters.
    J = np.asarray(jax.jacrev(log_burden)(lt))  # (N_SAMPLES, n_theta)
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

    fim = (J.T @ J) / (NOISE_SIGMA**2)
    eig = np.linalg.eigvalsh(fim)
    eig = np.clip(eig, 0.0, None)

    names = list(THETA_RANGES[model_id].keys())
    # Profile-likelihood CI half-width, via the marginal standard error:
    # se_i = sqrt((FIM^-1)_ii). Unbounded when the FIM is singular in that
    # direction, which is exactly the "flat" case we are looking for.
    try:
        cov = np.linalg.pinv(fim, rcond=1e-12)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(names), np.inf)

    # A column of J that is EXACTLY zero means the parameter has no effect on
    # the observable at all -- here, the drug parameters under an untreated
    # (u=0) run. The pseudo-inverse hands those a zero variance, which reads as
    # "perfectly identifiable" and is precisely backwards. They are
    # unidentifiable BY CONSTRUCTION and get their own class, so the count of
    # genuinely identifiable parameters is not silently inflated.
    sens = np.abs(J).max(axis=0)

    classes = {}
    for i, nm in enumerate(names):
        # se is in log-theta, so 1.96*se is a log-scale CI half-width.
        width_decades = float(2 * 1.96 * se[i] / np.log(10))
        if sens[i] <= 0.0:
            cls = "no-effect"
            width_decades = float("inf")
        elif not np.isfinite(width_decades) or width_decades > 3.0:
            cls = "flat"
        elif width_decades < 1.0:
            cls = "identifiable"
        else:
            cls = "weak"
        classes[nm] = {
            "ci_width_decades": (
                None if not np.isfinite(width_decades) else round(width_decades, 3)
            ),
            "max_abs_sensitivity": float(sens[i]),
            "class": cls,
        }

    ev_max, ev_min = float(eig.max()), float(eig.min())
    return {
        "eigenvalues": [float(x) for x in eig],
        "condition_number": float(ev_max / ev_min) if ev_min > 0 else float("inf"),
        "components": classes,
    }


# --- (2) separability + (3) collision --------------------------------------


def separability(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    X, y, raw = [], [], {}
    for k, fam in enumerate(FAMILIES):
        c = sample_cohort(fam, N_PER_FAMILY, seed=seed)
        b = add_noise(burdens(fam, c.thetas, c.y0s), rng)
        raw[fam] = b
        X.append(featurise(b))
        y.append(np.full(c.n, k))
        rejected = c.n_rejected
    X = np.concatenate(X)
    y = np.concatenate(y)

    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    split = (2 * len(y)) // 3
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    # 1-NN by Euclidean distance on the normalised shape.
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    pred = ytr[np.argmin(d2, axis=1)]

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    per_class = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)

    return {
        "seed": seed,
        "balanced_accuracy": float(per_class.mean()),
        "per_class_recall": {f: float(r) for f, r in zip(FAMILIES, per_class)},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(yte)),
        "n_rejected_last_family": int(rejected),
        "_raw": raw,
    }


def collisions(raw: dict) -> dict:
    """Fraction of cross-family pairs agreeing within relative L2 < 5%."""
    out = {}
    for i, a in enumerate(FAMILIES):
        for b in FAMILIES[i + 1 :]:
            A, B = featurise(raw[a]), featurise(raw[b])
            m = min(len(A), len(B), 150)  # capped: O(m^2) pairs
            A, B = A[:m], B[:m]
            num = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            den = np.maximum(np.sqrt((A**2).sum(-1))[:, None], 1e-12)
            rel = num / den
            out[f"{a}|{b}"] = float((rel < COLLISION_TOL).mean())
    return out


def main() -> int:
    print("=" * 72)
    print("PHASE 0 — identifiability + separability (burden-only)")
    print(f"  window {T1:.0f} d, {N_SAMPLES} samples, noise sigma={NOISE_SIGMA}")

    # (1) identifiability, on one representative patient per family
    ident = {}
    for fam in FAMILIES:
        c = sample_cohort(fam, 1, seed=0)
        ident[fam] = fisher_and_profiles(fam, c.thetas[0], c.y0s[0])
        classes = [v["class"] for v in ident[fam]["components"].values()]
        print(
            f"  {fam:16s} identifiable={classes.count('identifiable')} "
            f"weak={classes.count('weak')} flat={classes.count('flat')} "
            f"no-effect={classes.count('no-effect')}"
        )

    # (2) separability across seeds
    runs, raws = [], None
    for s in SEEDS:
        r = separability(s)
        raws = r.pop("_raw")
        runs.append(r)
        print(f"  seed {s}: balanced accuracy {r['balanced_accuracy']:.3f}")

    accs = [r["balanced_accuracy"] for r in runs]
    median_acc = float(np.median(accs))

    # (3) collisions, on the last seed's cohort
    coll = collisions(raws)
    worst_pair, worst = max(coll.items(), key=lambda kv: kv[1])

    passed = median_acc >= ACCURACY_GATE
    verdict = "PASS" if passed else "FAIL — reframe Q3"

    result = {
        "prereg": "adaptive_therapy/PREREG.md [2026-09-22]",
        "verdict": verdict,
        "gate": {
            "criterion": f"median balanced accuracy >= {ACCURACY_GATE}",
            "median_balanced_accuracy": median_acc,
            "range": [float(min(accs)), float(max(accs))],
            "per_seed": accs,
        },
        "window": {"t1_days": T1, "n_samples": N_SAMPLES, "noise_sigma": NOISE_SIGMA},
        "identifiability": ident,
        "separability": runs,
        "collisions": coll,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase0_identifiability.json").write_text(json.dumps(result, indent=2))

    print("-" * 72)
    print(
        f"  GATE: median balanced accuracy {median_acc:.3f} "
        f"(range {min(accs):.3f}-{max(accs):.3f}) vs {ACCURACY_GATE} -> {verdict}"
    )
    print(f"  worst colliding pair: {worst_pair} at {worst:.1%}")
    print(f"  -> {OUT / 'phase0_identifiability.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
