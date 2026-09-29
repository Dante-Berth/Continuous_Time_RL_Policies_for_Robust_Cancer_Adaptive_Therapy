"""ROADMAP W1: why did the ODE-RNN lose on Tier 1? A diagnostic, not a verdict.

Not pre-registered: no retraining, no new hypothesis, no gate. It reads the 12
saved Phase-2c policies (RESEARCH_LOG L424) and measures two things that decide
the paper's WORDING, never a criterion:

  Tier 0   the same test patients WITHOUT the Tier-1 shift (theta in range, y0
           as sampled). ODE ~ MLP here but worse on Tier 1 => its learned
           internal model hurts under shift. Worse on both => a training /
           capacity gap, not a robustness finding.
  Rule     how much of each arm's dose is explained by the CURRENT observation
           alone: accuracy of the per-bin majority dose over 20 quantile bins of
           obs. The MLP is memoryless, so it scores ~1 by construction and its
           exact rule is read off a grid of obs; the ODE-RNN's shortfall from 1
           is how much it doses from memory.

Tier 1 is re-evaluated with the gate's own eval key, so its numbers must match
phase2c_gate.json exactly -- a reproduction check on the loaded policies.

Run: JAX_PLATFORMS=cpu uv run python research/w1_diagnose.py
     -> research/outputs/w1_diagnose.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_gate as g  # noqa: E402  (sets the backend env before jax work)

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.eval.metrics import cumulative_exposure, paired_wilcoxon, ttp_running_nadir
from adaptive_therapy.policies.base import DOSE_LEVELS
from adaptive_therapy.rl import ppo

PHASE = "phase2c"
N_BINS = 20
OBS_GRID = np.linspace(-4.0, 1.0, 501)  # log(burden / baseline)


def tier0_cohort(n: int, seed: int):
    """The Tier-1 patients before the shift: same sample, theta and y0 as drawn."""
    c = sample_cohort(g.FAM, n, seed=seed + 1000, ranges=g.RANGES)
    return c.thetas, c.y0s


def load(arch: str, w: float, s: int):
    like = g.REGISTRATIONS[PHASE]["archs"][arch](jax.random.PRNGKey(0))
    path = g.OUT / f"{PHASE}_policies" / f"{arch}_w{w}_s{s}.eqx"
    return eqx.tree_deserialise_leaves(path, like)


def obs_explained(obs: np.ndarray, dose: np.ndarray) -> float:
    """Fraction of visits whose dose equals the majority dose of its obs bin."""
    o, d = obs.ravel(), dose.ravel()
    edges = np.unique(np.quantile(o, np.linspace(0, 1, N_BINS + 1)))
    b = np.clip(np.searchsorted(edges, o, side="right") - 1, 0, len(edges) - 2)
    hit = 0
    for i in np.unique(b):
        _, counts = np.unique(d[b == i], return_counts=True)
        hit += counts.max()
    return float(hit / d.size)


def mlp_rule(policy) -> list[dict]:
    """The MLP's exact greedy dose as a function of obs: its switch points."""
    a = np.asarray(
        jax.vmap(lambda o: jnp.argmax(policy(o, policy.initial_state(), 7.0)[0]))(
            jnp.asarray(OBS_GRID)
        )
    )
    segs, start = [], 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[start]:
            segs.append(
                {
                    "obs_from": float(OBS_GRID[start]),
                    "obs_to": float(OBS_GRID[i - 1]),
                    "dose": float(DOSE_LEVELS[a[start]]),
                }
            )
            start = i
    return segs


def evaluate(policy, th, y0, w, s):
    k_eval = jax.random.split(jax.random.PRNGKey(s), 3)[2]  # the gate's eval key
    ev = ppo.evaluate(
        policy, jnp.asarray(th), jnp.asarray(y0), k_eval, g.EP, RewardConfig(exposure_penalty=w)
    )
    burden, dose, obs = (np.asarray(ev[k]) for k in ("burden", "dose", "obs"))
    ttp = ttp_running_nadir(burden, g.TS)
    return {
        "ttp_per_patient": ttp.days.tolist(),
        "ttp_median_capped": ttp.median_iqr_capped()[0],
        "censoring_rate": ttp.censoring_rate,
        "exposure_median": float(np.median(cumulative_exposure(dose, g.TS))),
        "dose_explained_by_obs": obs_explained(obs, dose),
    }


def main() -> int:
    gate = json.loads((g.OUT / f"{PHASE}_gate.json").read_text())
    ref = {(r["arch"], r["exposure_weight"], r["seed"]): r for r in gate["runs"]}
    cells, rules = [], {}
    for w in g.EXPOSURE_WEIGHTS:
        for arch in g.REGISTRATIONS[PHASE]["archs"]:
            for s in g.SEEDS:
                pol = load(arch, w, s)
                row = {"arch": arch, "w": w, "seed": s}
                for tier, cohort in (("tier0", tier0_cohort), ("tier1", g.tier1_cohort)):
                    row[tier] = evaluate(pol, *cohort(g.N_TEST, s), w, s)
                r1 = ref[(arch, w, s)]
                row["tier1_reproduces_gate"] = bool(
                    np.array_equal(row["tier1"]["ttp_per_patient"], r1["ttp_per_patient"])
                )
                if arch == "mlp":
                    rules[f"w{w}_s{s}"] = mlp_rule(pol)
                cells.append(row)
                print(
                    f"w={w:<4} {arch:8s} s{s}  "
                    + "  ".join(
                        f"{t}: TTP {row[t]['ttp_median_capped']:5.0f} cens {row[t]['censoring_rate']:.0%} "
                        f"expo {row[t]['exposure_median']:5.0f} obs-expl {row[t]['dose_explained_by_obs']:.2f}"
                        for t in ("tier0", "tier1")
                    )
                    + f"  repro={row['tier1_reproduces_gate']}"
                )

    paired = {}
    for w in g.EXPOSURE_WEIGHTS:
        for tier in ("tier0", "tier1"):
            o, m = (
                np.concatenate(
                    [c[tier]["ttp_per_patient"] for c in cells if c["arch"] == a and c["w"] == w]
                )
                for a in ("ode_rnn", "mlp")
            )
            paired[f"w{w}_{tier}"] = {
                "ode_later": int((o > m).sum()),
                "ode_earlier": int((o < m).sum()),
                "wilcoxon_two_sided": paired_wilcoxon(o, m),
            }
            p = paired[f"w{w}_{tier}"]
            print(
                f"w={w} {tier}: ODE later/earlier {p['ode_later']}/{p['ode_earlier']} "
                f"p={p['wilcoxon_two_sided']['p_value']:.2g}"
            )
    for k, segs in rules.items():
        print(
            f"MLP rule {k}: "
            + " | ".join(f"[{x['obs_from']:+.2f},{x['obs_to']:+.2f}]→{x['dose']}" for x in segs)
        )

    out = {
        "source": f"{PHASE}_policies (L424)",
        "cells": cells,
        "paired": paired,
        "mlp_rules": rules,
    }
    (g.OUT / "w1_diagnose.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
