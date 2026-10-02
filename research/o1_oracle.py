"""O1: how much U is there to gain with full information, measured without a trainer?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-02] O1 (afa0aa7), before
any O1 search ran. At regular visits the simulator is deterministic given the
doses, so the full-information optimum is the best dose sequence per patient;
any sequence found bounds it from below.

  (i)   tuned threshold: best ThresholdPolicy level 0.30-1.50 (+ none, MTD) per patient
  (ii)  cross-entropy search over {0, 1/2, 1}^52, warm-started at (i)
  Primary  Delta_O = U(oracle) - U(MLP), S1 Part A Tier 0, seed+patient CI:
           lo > +10 d => headroom; hi < +10 d => no headroom found; else inconclusive.

Run: JAX_PLATFORMS=cpu uv run python research/o1_oracle.py
     -> research/outputs/o1_oracle.{json,log}, o1_shards/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets g.RANGES = "strobl2021"; backend env first)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.eval.metrics import (  # noqa: E402
    burden_integral,
    cumulative_exposure,
    ttp_running_nadir,
)
from adaptive_therapy.policies.base import N_ACTIONS  # noqa: E402
from adaptive_therapy.policies.rules import (  # noqa: E402
    RulePolicy,
    SequencePolicy,
    ThresholdPolicy,
)
from adaptive_therapy.rl.rollout import batch_rollout, rollout  # noqa: E402

w4, g = s1.w4, s1.g
SEEDS = s1.SEEDS_A
SHARDS = g.OUT / "o1_shards"
RESULT = g.OUT / "o1_oracle.json"
EP = replace(g.EP, obs_noise=0.0)  # the oracle sees the true state: no noise
LEVELS = np.round(np.arange(0.30, 1.5001, 0.05), 2)
POP, ITERS, ELITE, SMOOTH, WARM_P = 128, 25, 13, 0.7, 0.8  # elite = 10% of 128
HEADROOM_D = 10.0
KEY = jax.random.PRNGKey(0)


def endpoints(burden, dose) -> dict:
    b, d = np.asarray(burden), np.asarray(dose)
    ttp = ttp_running_nadir(b, g.TS)
    return {
        "ttp_capped": ttp.days.tolist(),
        "censored": ttp.censored.tolist(),
        "exposure": cumulative_exposure(d, g.TS).tolist(),
        "burden_integral": burden_integral(b, g.TS).tolist(),
    }


def u_of(e: dict, w: float) -> np.ndarray:
    return np.minimum(np.asarray(e["ttp_capped"]), w4.HORIZON) - w * np.asarray(e["exposure"])


def objective_fn(w: float):
    """Search objective: V2's online survival (TTP + 28 d) - w * dose; burden term 0."""
    rw = RewardConfig(survival_until="progression", burden_penalty=0.0, exposure_penalty=w)

    @jax.jit
    def f(acts, th, y0):
        keys = jax.random.split(KEY, acts.shape[0])
        tr = jax.vmap(lambda a, t, y, k: rollout(SequencePolicy(a), t, y, k, EP, rw, greedy=True))(
            acts, th, y0, keys
        )
        return tr["reward"].sum(1)

    return f


@jax.jit
def play(acts, th, y0):
    keys = jax.random.split(KEY, acts.shape[0])
    tr = jax.vmap(
        lambda a, t, y, k: rollout(SequencePolicy(a), t, y, k, EP, RewardConfig(), greedy=True)
    )(acts, th, y0, keys)
    return tr["burden"], tr["dose"]


def rule_trajectories(th, y0) -> list[dict]:
    """(i)'s candidates: every threshold level, none and MTD (w-free trajectories)."""
    pols = [ThresholdPolicy(float(x)) for x in LEVELS] + [RulePolicy("none"), RulePolicy("MTD")]
    out = []
    for p in pols:
        tr = batch_rollout(p, th, y0, KEY, EP, RewardConfig(), greedy=True)
        out.append({"actions": np.asarray(tr["action"]), **endpoints(tr["burden"], tr["dose"])})
    return out


def cem(f, warm, th, y0, key):
    """Cross-entropy over per-visit dose distributions, all patients at once."""
    n, t = warm.shape
    probs = jnp.full((n, t, N_ACTIONS), (1.0 - WARM_P) / (N_ACTIONS - 1))
    probs = jnp.where(jax.nn.one_hot(warm, N_ACTIONS) > 0, WARM_P, probs)
    best_seq, best_obj = jnp.asarray(warm), f(jnp.asarray(warm), th, y0)
    th_r, y0_r = jnp.repeat(th, POP, 0), jnp.repeat(y0, POP, 0)
    trace = [float(best_obj.mean())]
    for _ in range(ITERS):
        key, k = jax.random.split(key)
        samp = jax.random.categorical(k, jnp.log(probs)[:, None], axis=-1, shape=(n, POP, t))
        obj = f(samp.reshape(n * POP, t), th_r, y0_r).reshape(n, POP)
        top = jnp.argsort(-obj, axis=1)[:, :ELITE]
        elites = jnp.take_along_axis(samp, top[:, :, None], axis=1)
        new = jax.nn.one_hot(elites, N_ACTIONS).mean(1)
        probs = SMOOTH * new + (1.0 - SMOOTH) * probs
        i = jnp.argmax(obj, axis=1)
        cand, cand_obj = samp[jnp.arange(n), i], obj[jnp.arange(n), i]
        better = cand_obj > best_obj
        best_seq = jnp.where(better[:, None], cand, best_seq)
        best_obj = jnp.where(better, cand_obj, best_obj)
        trace.append(float(best_obj.mean()))
    return np.asarray(best_seq), trace


def worker(seed: int) -> dict:
    th, y0 = (jnp.asarray(x) for x in w4.tier0_cohort(g.N_TEST, seed))
    rules = rule_trajectories(th, y0)
    out = {"seed": seed, "w": {}}
    for w in w4.WEIGHTS:
        us = np.stack([u_of(r, w) for r in rules])  # (n_rules, N)
        pick = us.argmax(0)
        tuned = {
            k: [rules[j][k][i] for i, j in enumerate(pick)]
            for k in ("ttp_capped", "censored", "exposure", "burden_integral")
        }
        warm = np.stack([rules[j]["actions"][i] for i, j in enumerate(pick)])
        f = objective_fn(w)
        seq, trace = cem(f, warm, th, y0, jax.random.fold_in(KEY, seed * 100 + int(w * 100)))
        oracle = endpoints(*play(jnp.asarray(seq), th, y0))
        out["w"][str(w)] = {
            "oracle": oracle,
            "tuned": tuned,
            "tuned_level": [
                float(LEVELS[j]) if j < len(LEVELS) else ("none", "MTD")[j - len(LEVELS)]
                for j in pick
            ],
            "trace": trace,
            "oracle_below_tuned": float(np.mean(u_of(oracle, w) < u_of(tuned, w) - 1e-9)),
        }
    return out


def spawn(seed: int) -> dict:
    shard = SHARDS / f"s{seed}.json"
    if not shard.exists():  # resume: a finished seed is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done seed {seed}", flush=True)
    return json.loads(shard.read_text())


def reading(ci) -> str:
    lo, hi = ci
    if lo > HEADROOM_D:
        return "headroom: full information is worth > 10 d; the MLP's limit is learning"
    if hi < HEADROOM_D:
        return "no headroom found: LAW L1's mechanism gets direct support (lower bound)"
    return "inconclusive"


def analyse(shards: list[dict]) -> dict:
    cells = []
    for sh in shards:
        for w in w4.WEIGHTS:
            d = sh["w"][str(w)]
            for arch in ("oracle", "tuned"):
                cells.append({"arch": arch, "w": w, "seed": sh["seed"], "tier0": d[arch]})
    for w in w4.WEIGHTS:
        for s in SEEDS:
            c = json.loads((s1.SHARDS / f"a_mlp_w{w}_s{s}.json").read_text())
            cells.append({"arch": "mlp", "w": w, "seed": s, "tier0": c["tier0"]})
    rules = [json.loads(p.read_text()) for p in sorted(s1.SHARDS.glob("ra_*.json"))]
    valid = {
        (c["arch"], c["w"], c["seed"]): bool(
            np.isfinite(c["tier0"]["ttp_capped"]).all()
            and np.isfinite(c["tier0"]["exposure"]).all()
        )
        for c in cells
    }

    def cmp(a, b):
        return w4.comparison(cells, valid, a, b, "level", SEEDS, ("tier0",))

    res = {
        "prereg": "adaptive_therapy/PREREG.md O1 [2026-10-02]",
        "nonfinite": [str(k) for k, ok in valid.items() if not ok],
        "primary": cmp("oracle", "mlp"),
        "secondary_tuned": cmp("tuned", "mlp"),
        "oracle_vs_tuned": cmp("oracle", "tuned"),
        "table": w4.table(cells, valid, rules, ("oracle", "tuned", "mlp"), SEEDS, ("tier0",)),
    }
    res["reading"] = reading(res["primary"]["ci95"])
    d = w4.paired(cells, valid, "oracle", "mlp", "level", SEEDS, ("tier0",))
    res["share_gain_gt_10d"] = float(np.nanmean(d > HEADROOM_D))
    tr = np.array([sh["w"][str(w)]["trace"] for sh in shards for w in w4.WEIGHTS])
    res["search_last5_gain"] = float((tr[:, -1] - tr[:, -6]).mean())
    res["search_total_gain"] = float((tr[:, -1] - tr[:, 0]).mean())
    res["oracle_below_tuned"] = float(
        np.mean([sh["w"][str(w)]["oracle_below_tuned"] for sh in shards for w in w4.WEIGHTS])
    )
    return res


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as ex:  # V2 holds most cores (PREREG O1 compute note)
        shards = list(ex.map(spawn, SEEDS))
    res = analyse(shards)
    print(f"O1 primary  oracle − MLP  {s1.ci(res['primary'])}")
    print(f"secondary   tuned − MLP   {s1.ci(res['secondary_tuned'])}")
    print(f"            oracle − tuned {s1.ci(res['oracle_vs_tuned'])}")
    print(f"per w: {json.dumps({k: round(v, 1) for k, v in res['primary']['per_w_p'].items()})}")
    print(f"share of patient-cells gaining > 10 d: {res['share_gain_gt_10d']:.2f}")
    print(
        f"search gain over warm start {res['search_total_gain']:+.1f}; last 5 iters "
        f"{res['search_last5_gain']:+.2f}; oracle below tuned on U in {res['oracle_below_tuned']:.1%}"
    )
    for name in ("oracle", "tuned", "mlp", "thr0.8", "MTD"):
        for w in (0.05, 0.4):
            t = res["table"].get(f"{name}|{w}|tier0")
            if t:
                print(
                    f"  w{w} {name:7s} U {t['U']:6.1f}  TTP {t['ttp_median']:.0f} "
                    f"[{t['ttp_iqr'][0]:.0f}–{t['ttp_iqr'][1]:.0f}]  cens {t['censoring']:.0%}  "
                    f"exp {t['exposure_median']:.0f}  burden {t['burden_integral_median']:.3g}"
                )
    print(f"READING: {res['reading']}")
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        seed = int(sys.argv[2])
        (SHARDS / f"s{seed}.json").write_text(json.dumps(worker(seed)))
        raise SystemExit(0)
    raise SystemExit(main())
