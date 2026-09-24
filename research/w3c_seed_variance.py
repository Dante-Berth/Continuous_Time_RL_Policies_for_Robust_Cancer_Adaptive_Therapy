"""W3c re-analysis: how much of its CI is training variance? Not pre-registered.

W3c's CIs resample patients with each trained policy held fixed. Found while
building W4 (2026-09-25): retraining one ODE-RNN cell under a float perturbation
of ~2e-7 (the W4 flow's traced gap) moved its Tier-0 mean U by >100 d. Two parts:

  bootstrap  W3c's Deltas, same valid cells, with a CI resampling seeds AND
             patients (w4_irregular.delta_ci) beside W3c's patient-only CI.
  retrain    ode_rnn and lstm w0.2 s7 retrained under the current code at
             drop 0: mean Tier-0 U vs the W3c shard. The LSTM has no flow, so
             it should retrain bit-identically; the ODE-RNN measures chaos.

Descriptive only: W3c's registered verdicts are not re-graded.

Run: JAX_PLATFORMS=cpu uv run python research/w3c_seed_variance.py
     -> research/outputs/w3c_seed_variance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0], "w3c"]  # w3_frontier reads its registration from argv
import w3_frontier as wf  # noqa: E402
import w4_irregular as w4  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import cumulative_exposure, ttp_running_nadir  # noqa: E402
from adaptive_therapy.rl import ppo, pqn  # noqa: E402

g = wf.g
RETRAIN = (("ode_rnn", 0.2, 7), ("lstm", 0.2, 7))


def shard(a, w, s) -> dict:
    return json.loads((g.OUT / "w3c_shards" / f"{a}_w{w}_s{s}.json").read_text())


def u(tier: dict, w: float) -> np.ndarray:
    return np.asarray(tier["ttp_capped"]) - w * np.asarray(tier["exposure"])


def bootstrap() -> dict:
    excluded = set(json.loads((g.OUT / "w3c_utility.json").read_text())["excluded"])
    ok = {
        (a, w, s): f"{a}_w{w}_s{s}" not in excluded
        for a in wf.ARCHS
        for w in wf.WEIGHTS
        for s in wf.SEEDS
    }
    cells = {k: shard(*k) for k in ok}
    out = {
        "per_cell_tier1_mean_U": {
            f"{a}_w{w}_s{s}": float(u(c["tier1"], w).mean()) for (a, w, s), c in cells.items()
        }
    }
    for a, b in (("lstm", "mlp"), ("ode_rnn", "lstm")):
        for t in ("tier1", "tier0"):
            d = np.full((len(wf.WEIGHTS), 1, len(wf.SEEDS), g.N_TEST), np.nan)
            for i, w in enumerate(wf.WEIGHTS):
                for k, s in enumerate(wf.SEEDS):
                    if ok[(a, w, s)] and ok[(b, w, s)]:
                        d[i, 0, k] = u(cells[(a, w, s)][t], w) - u(cells[(b, w, s)][t], w)
            out[f"{a}-{b}|{t}"] = w4.delta_ci(d, (t,))
    return out


def retrain(arch: str, w: float, s: int) -> dict:
    k_init, k_train, k_eval = jax.random.split(jax.random.PRNGKey(s), 3)
    c = sample_cohort(g.FAM, g.N_TRAIN, seed=s)
    rw = RewardConfig(exposure_penalty=w)
    pol, _ = pqn.train(
        wf.ARCHS[arch](k_init),
        jnp.asarray(c.thetas),
        jnp.asarray(c.y0s),
        k_train,
        pqn.PQNConfig(),
        g.EP,
        rw,
    )
    th, y0 = wf.TIERS["tier0"](g.N_TEST, s)
    ev = ppo.evaluate(pol, jnp.asarray(th), jnp.asarray(y0), k_eval, g.EP, rw)
    new = ttp_running_nadir(np.asarray(ev["burden"]), g.TS).days - w * cumulative_exposure(
        np.asarray(ev["dose"]), g.TS
    )
    ref = u(shard(arch, w, s)["tier0"], w)
    return {
        "mean_U_retrained": float(new.mean()),
        "mean_U_w3c": float(ref.mean()),
        "identical": bool(np.array_equal(new, ref)),
        "n_patients_differing": int((new != ref).sum()),
    }


def main() -> int:
    res = {"bootstrap": bootstrap()}
    for k, v in res["bootstrap"].items():
        if "|" in k:
            print(
                f"{k:18s} Δ {v['delta']:+6.1f}  patients {np.round(v['ci95_patients'], 1)}"
                f"  seeds+patients {np.round(v['ci95'], 1)}  per seed {np.round(v['per_seed'], 1)}"
            )
    res["retrain"] = {f"{a}_w{w}_s{s}": retrain(a, w, s) for a, w, s in RETRAIN}
    for k, v in res["retrain"].items():
        print(k, v)
    (g.OUT / "w3c_seed_variance.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
