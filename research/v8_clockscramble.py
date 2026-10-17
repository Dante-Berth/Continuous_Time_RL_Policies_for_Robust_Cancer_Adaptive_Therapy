"""V8: scramble the clock of V7's trained positional MLPs (no retraining).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-17] V8, before any cell
ran. Tests V7's post-hoc reading (L1414) that the clock lets PQN fit a
time-indexed schedule: if so, a clock that carries no information must cost U.

  intact    V7's policy, unchanged. Must reproduce V7's shard to 1e-6 or the run is void.
  frozen    clock input = 178.5 d at every visit.   PRIMARY: frozen - intact, Tier 1.
  reversed  clock input = 357 d - elapsed.          Consistency check, not a gate.

Run: JAX_PLATFORMS=cpu uv run python research/v8_clockscramble.py
     -> research/outputs/v8_clockscramble.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_strobl as s1  # noqa: E402  (sets Strobl ranges; backend env first)
from v1_review import median_ci  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from adaptive_therapy.policies.mlp import MLPPolicy, ScrambledClock  # noqa: E402

w4, g = s1.w4, s1.g
MODES = ("intact", "frozen", "reversed")
V7_POLICIES = g.OUT / "v7_policies" / "a"
V7_SHARDS = g.OUT / "v7_shards"
SHARDS = g.OUT / "v8_shards"
GATE = 10.0  # days; the program's gate, as registered
TOL = 1e-6


def load_pos(w: float, seed: int) -> MLPPolicy:
    like = MLPPolicy(jax.random.PRNGKey(0), width=386, layer_norm=True, clock=True)
    return eqx.tree_deserialise_leaves(V7_POLICIES / f"pos_mlp_w{w}_s{seed}.eqx", like)


def cell(mode: str, w: float, seed: int) -> dict:
    policy = ScrambledClock(load_pos(w, seed), mode)
    return {"arch": mode, "w": w, "seed": seed, **s1.tiers_eval(policy, seed, w)}


def check_intact(c: dict) -> None:
    """Void the run if the wrapper does not reproduce V7's own evaluation."""
    ref = json.loads((V7_SHARDS / f"a_pos_mlp_w{c['w']}_s{c['seed']}.json").read_text())
    for tier in s1.TIERS:
        for k in ("ttp_capped", "exposure", "burden_integral"):
            if not np.allclose(c[tier][k], ref[tier][k], atol=TOL, rtol=0):
                raise SystemExit(f"VOID: intact != V7 shard ({tier} {k} w={c['w']} s={c['seed']})")


def describe(cells: list[dict], mode: str, tier: str) -> dict:
    cs = [c for c in cells if c["arch"] == mode]
    pool = lambda k: np.concatenate([np.asarray(c[tier][k]) for c in cs])  # noqa: E731
    q = lambda x: [float(v) for v in np.percentile(x, [25, 50, 75])]  # noqa: E731
    return {
        "ttp_capped_q25_50_75": q(pool("ttp_capped")),
        "exposure_q25_50_75": q(pool("exposure")),
        "burden_integral_q25_50_75": q(pool("burden_integral")),
        "closed_loop_mean": float(np.mean([c[f"closed_loop_{tier}"] for c in cs])),
    }


def wilcoxon_p(d: np.ndarray) -> float:
    x = d[np.isfinite(d)].ravel()
    return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0


def analyse(cells: list[dict]) -> dict:
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in cells}
    res = {"prereg": "adaptive_therapy/PREREG.md V8 [2026-10-17]", "arms": {}, "vs_intact": {}}
    res["nonfinite"] = [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok]
    for tier in s1.TIERS:
        for m in MODES:
            res["arms"][f"{m}_{tier}"] = describe(cells, m, tier)
        for m in ("frozen", "reversed"):
            d = w4.paired(cells, valid, m, "intact", "level", s1.SEEDS_A, (tier,))
            res["vs_intact"][f"{m}_{tier}"] = {
                **w4.delta_ci(d, (tier,)),
                "median": median_ci(d),
                "wilcoxon_p_pooled": wilcoxon_p(d),
            }
    lo, hi = res["vs_intact"]["frozen_tier1"]["ci95"]
    res["primary_frozen_tier1"] = (
        "uses the clock as a schedule" if hi < -GATE
        else "ignores it" if -GATE < lo and hi < GATE
        else "inconclusive"
    )  # fmt: skip
    return res


def ci(v: dict) -> str:
    m = v["median"]
    return (
        f"{v['delta']:+6.1f} [{v['ci95'][0]:+6.1f}, {v['ci95'][1]:+6.1f}]"
        f"  median {m['delta']:+.1f} [{m['ci95'][0]:+.1f}, {m['ci95'][1]:+.1f}]"
        f"  wilcoxon p {v['wilcoxon_p_pooled']:.2g}"
    )


def worker(w: float, seed: int) -> None:
    """One (w, seed) in its own process: JAX's JIT mappings are never freed
    (L207), so a long single process crosses vm.max_map_count."""
    out = []
    for mode in MODES:
        c = cell(mode, w, seed)
        if mode == "intact":
            check_intact(c)
        out.append(c)
    (SHARDS / f"w{w}_s{seed}.json").write_text(json.dumps(out))


def spawn(w: float, seed: int) -> list[dict]:
    shard = SHARDS / f"w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(w, s) for w in w4.WEIGHTS for s in s1.SEEDS_A]
    with ThreadPoolExecutor(4) as ex:
        cells = [c for out in ex.map(lambda j: spawn(*j), jobs) for c in out]
    res = analyse(cells)
    (g.OUT / "v8_clockscramble.json").write_text(json.dumps(res, indent=2))
    print(f"nonfinite {res['nonfinite']}")
    for k, v in res["vs_intact"].items():
        print(f"  {k:16s} - intact {ci(v)}")
    for k, v in res["arms"].items():
        print(f"  {k:16s} {v}")
    print(f"PRIMARY frozen - intact, Tier 1: {res['primary_frozen_tier1']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]), int(sys.argv[3]))
    else:
        sys.exit(main())
