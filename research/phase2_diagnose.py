"""Why did Phase 2's ODE-RNN arm go open-loop? A diagnostic, not a gate.

Not pre-registered: it produces no comparison and no verdict. RESEARCH_LOG
L312 found the ODE-RNN giving all 200 test patients one identical exposure in
4/6 cells. This measures, over training, for BOTH arms under the Phase 2
trainer as it stood:

  entropy      mean policy entropy on visited states (max ln 3 = 1.10 nats)
  obs_dep      observation_dependence(): 0 = open-loop schedule
  mean_dose

at updates {0, 10, 30, 100, 300}. train() is deterministic in its key, so a
shorter run is an exact prefix of a longer one and each checkpoint is a
from-scratch retrain rather than a saved snapshot.

Deliberately reports NO TTP or exposure: it is run on the fixed trainer before
Phase 2b is registered, and must not be able to inform that registration's
outcome -- only whether both arms learn feedback policies at all.

Run: JAX_PLATFORMS=cpu uv run python research/phase2_diagnose.py [tag]
     -> research/outputs/phase2_diagnose[_tag].json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_gate as g  # noqa: E402  (sets the backend env before jax work)

from adaptive_therapy.envs.reward import RewardConfig
from adaptive_therapy.envs.sampling import sample_cohort
from adaptive_therapy.rl.ppo import PPOConfig, train
from adaptive_therapy.rl.rollout import observation_dependence, replay, rollout

OUT = Path(__file__).resolve().parent / "outputs"
CELLS = ((0.05, 0), (0.20, 1))  # both collapsed for the ODE arm in run 2
CHECKPOINTS = (0, 10, 30, 100, 300)


def probe(policy, th, y0, key, rw) -> dict:
    """Greedy rollouts on the Tier-1 test cohort, then replay the observations."""
    keys = jax.random.split(key, th.shape[0])
    traj = jax.vmap(lambda t, y, k: rollout(policy, t, y, k, g.EP, rw, greedy=True))(th, y0, keys)
    logits = jax.vmap(lambda o: replay(policy, o, g.EP)[0])(traj["obs"])
    p = jax.nn.softmax(logits, axis=-1)
    ent = -(p * jnp.log(jnp.maximum(p, 1e-10))).sum(-1).mean()
    return {
        "entropy": float(ent),
        "obs_dep": observation_dependence(policy, traj["obs"], g.EP, key),
        "mean_dose": float(traj["dose"].mean()),
    }


def main() -> int:
    report = {"note": "diagnostic, not pre-registered", "cells": []}
    for weight, seed in CELLS:
        train_c = sample_cohort(g.FAM, g.N_TRAIN, seed=seed)
        th, y0 = (jnp.asarray(a) for a in g.tier1_cohort(g.N_TEST, seed))
        rw = RewardConfig(exposure_penalty=weight)
        for arch in g.ARCHS:
            k_init, k_train, k_eval = jax.random.split(jax.random.PRNGKey(seed), 3)
            rows = []
            for n in CHECKPOINTS:
                pol = g.ARCHS[arch](k_init)
                if n:
                    pol, _ = train(
                        pol,
                        jnp.asarray(train_c.thetas),
                        jnp.asarray(train_c.y0s),
                        k_train,
                        PPOConfig(n_updates=n),
                        g.EP,
                        rw,
                    )
                row = {"updates": n, **probe(pol, th, y0, k_eval, rw)}
                rows.append(row)
                print(
                    f"  w={weight:<4} seed {seed} {arch:8s} upd {n:3d}: "
                    f"entropy {row['entropy']:.3f}  obs_dep {row['obs_dep']:.3f}  "
                    f"dose {row['mean_dose']:.2f}",
                    flush=True,
                )
            report["cells"].append({"weight": weight, "seed": seed, "arch": arch, "rows": rows})

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""
    (OUT / f"phase2_diagnose{tag}.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
