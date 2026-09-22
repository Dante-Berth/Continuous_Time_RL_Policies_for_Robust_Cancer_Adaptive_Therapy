"""Phase 1 deliverable: per-family untreated trajectories + cohort report.

Not a gateable research result (no policy, no comparison), so it carries no
/prereg entry. It produces the PROJECT.md 11 Phase-1 artifact -- the
per-family untreated-trajectory plots -- and the cohort composition numbers
that Phase 2 needs: how many draws survive the gate, why the rest do not, and
how many carry a fast transient.

Run: uv run python research/phase1_cohorts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from adaptive_therapy.envs.base import SolverConfig, observe
from adaptive_therapy.envs.cohort import solve_cohort
from adaptive_therapy.envs.gate import bucket_by_stiffness, untreated_sanity_gate
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.eval.metrics import ttp_from_baseline, ttp_running_nadir

OUT = Path(__file__).resolve().parent / "outputs"
FAMILIES = ("lotka_volterra", "hahnfeldt", "kuznetsov")
N = 200
CFG = SolverConfig(t0=0.0, t1=365.0, dt0=0.05, n_save=53)


def main() -> int:
    ts = np.linspace(CFG.t0, CFG.t1, CFG.n_save)
    report, curves = {}, {}

    for fam in FAMILIES:
        c = sample_cohort(fam, N, seed=0)
        g = untreated_sanity_gate(fam, c.thetas, c.y0s, CFG)
        buckets = bucket_by_stiffness(c.thetas, c.y0s, g)

        th, y0 = c.thetas[g.keep], c.y0s[g.keep]
        burden = np.asarray(
            observe(solve_cohort(fam, jnp.asarray(th), jnp.asarray(y0), cfg=CFG), fam)
        )
        curves[fam] = burden

        # Untreated TTP: the control arm every policy must beat. Both rules,
        # so the sensitivity of the comparison to the rule is on the record
        # from the start rather than discovered later.
        nadir = ttp_running_nadir(burden, ts)
        base = ttp_from_baseline(burden, ts)
        nm, nq1, nq3 = nadir.median_iqr()
        bm, _, _ = base.median_iqr()

        report[fam] = {
            "gate": g.summary(),
            "n_bulk": int(len(buckets["bulk"][0])),
            "n_fast_transient": int(len(buckets["stiff"][0])),
            "untreated_ttp_running_nadir": {
                "median": nm,
                "iqr": [nq1, nq3],
                "censoring_rate": nadir.censoring_rate,
            },
            "untreated_ttp_from_baseline_secondary": {
                "median": bm,
                "censoring_rate": base.censoring_rate,
            },
            "log_growth_median": float(
                np.median(
                    np.log(np.maximum(burden[:, -1], 1e-30))
                    - np.log(np.maximum(burden[:, 0], 1e-30))
                )
            ),
        }

        print(
            f"  {fam:16s} kept {g.n_kept:3d}/{N}  fast-transient "
            f"{report[fam]['n_fast_transient']:3d}  untreated TTP "
            f"{nm:6.1f} d [{nq1:.0f}-{nq3:.0f}]  censored {nadir.censoring_rate:.0%}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_cohorts.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, fam in zip(axes, FAMILIES):
            b = curves[fam]
            for i in range(min(40, b.shape[0])):
                ax.plot(ts, np.maximum(b[i], 1e-30), lw=0.6, alpha=0.45)
            ax.set_yscale("log")
            ax.set_title(fam.replace("_", " "))
            ax.set_xlabel("days")
        axes[0].set_ylabel("tumour burden (log)")
        fig.suptitle("Untreated virtual-patient trajectories (n<=40 shown per family)")
        fig.tight_layout()
        fig.savefig(OUT / "phase1_untreated_trajectories.png", dpi=130)
        print(f"  -> {OUT / 'phase1_untreated_trajectories.png'}")
    except ImportError:
        print("  (matplotlib absent; plots skipped -- install the 'viz' extra)")

    print(f"  -> {OUT / 'phase1_cohorts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
