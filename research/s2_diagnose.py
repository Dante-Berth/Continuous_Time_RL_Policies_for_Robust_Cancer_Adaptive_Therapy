"""S2 diagnostic (NOT pre-registered): why is the full-information reference worse?

Saved Part-A policies only, no retraining. Per cell: mean episode return under
the cell's own reward, and mean U, on the Tier-0 test cohort and on the cell's
training cohort (same eval key). Reference - MLP, averaged over the 40 cells:
  misalignment  more reward but less U (the reward has no TTP term, L739)
  overfitting   better on its training patients, worse on test
  optimisation  less reward on both

Run: JAX_PLATFORMS=cpu uv run python research/s2_diagnose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_headroom as s2  # noqa: E402  (sets g.RANGES = "strobl2021")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import cumulative_exposure, ttp_running_nadir  # noqa: E402
from adaptive_therapy.rl import ppo  # noqa: E402

s1, w4, g = s2.s1, s2.w4, s2.g


def score(pol, ep, th, y0, key, w) -> dict:
    ev = ppo.evaluate(pol, th, y0, key, ep, RewardConfig(exposure_penalty=w))
    b, dose = np.asarray(ev["burden"]), np.asarray(ev["dose"])
    ttp = np.minimum(ttp_running_nadir(b, g.TS).days, w4.HORIZON)
    u = ttp - w * cumulative_exposure(dose, g.TS)
    return {"return": float(np.asarray(ev["reward"]).sum(1).mean()), "U": float(u.mean())}


def main() -> int:
    like = {"mlp": w4.ARCHS["mlp"](jax.random.PRNGKey(0))}
    like |= {o: s2.make(o)(jax.random.PRNGKey(0)) for o in s2.ORACLES}
    arms = [("mlp", s1.POLICIES / "a", g.EP)]
    arms += [(o, s2.POLICIES / "a", s2.ep_for(o)) for o in s2.ORACLES]
    rows = []
    for seed in s1.SEEDS_A:
        tr = sample_cohort(g.FAM, g.N_TRAIN, seed=seed, ranges=g.RANGES)
        coh = {
            "test": tuple(jnp.asarray(x) for x in w4.tier0_cohort(g.N_TEST, seed)),
            "train": (jnp.asarray(tr.thetas), jnp.asarray(tr.y0s)),
        }
        for w in w4.WEIGHTS:
            row = {"seed": seed, "w": w}
            for arch, d, ep in arms:
                pol = eqx.tree_deserialise_leaves(d / f"{arch}_w{w}_s{seed}.eqx", like[arch])
                for name, (th, y0) in coh.items():
                    row[f"{arch}|{name}"] = score(pol, ep, th, y0, w4.eval_key(seed), w)
            rows.append(row)
    out = {"rows": rows, "delta": {}}
    for o in s2.ORACLES:
        for name in ("test", "train"):
            for stat in ("return", "U"):
                d = np.array([r[f"{o}|{name}"][stat] - r[f"mlp|{name}"][stat] for r in rows])
                out["delta"][f"{o}-mlp|{name}|{stat}"] = [float(d.mean()), int((d > 0).sum())]
    for k, (m, pos) in out["delta"].items():
        print(f"{k:28s} {m:+8.2f}   positive in {pos}/{len(rows)} cells")
    (g.OUT / "s2_diagnose.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
