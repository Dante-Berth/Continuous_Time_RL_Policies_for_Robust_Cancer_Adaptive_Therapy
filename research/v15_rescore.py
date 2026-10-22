"""V15 stage 1: re-score the reference-vs-MLP gate on the burden integral, from saved shards.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-22] V15, before this ran.
No training: reads the per-patient arrays already in v12/v13/v14 shards.
Endpoint: ln(B_ref / B_mlp) per matched patient (negative = reference smaller),
seed+patient CI as every other gate; guard: exposure delta lower CI not > +5 d.

Run: JAX_PLATFORMS=cpu uv run python research/v15_rescore.py -> research/outputs/v15_rescore.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v12_ppo as v12  # noqa: E402  (imports s1_strobl: backend env first)

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

s1, w4, g = v12.s1, v12.w4, v12.g
# trainer -> (shard dir, reference arm, MLP arm); V13 and V14 are the criteria, V12 is a check
TRAINERS = {
    "v13_r2d2": ("v13_shards", "r2d2_ref", "r2d2_mlp"),
    "v14_es": ("v14_shards", "es_ref", "es_mlp"),
    "v12_ppo": ("v12_shards", "ppo_ref", "ppo_mlp"),
}
EXPOSURE_GUARD = 5.0  # d, registered
EPS = 1e-10


def load(shard_dir: str, arms: tuple[str, ...]) -> list[dict]:
    d = g.OUT / shard_dir
    return [
        json.loads(p.read_text())
        for a in arms
        for w in w4.WEIGHTS
        for sd in s1.SEEDS_A
        for p in [d / f"{a}_w{w}_s{sd}.json"]
    ]


def paired(cells, a: str, b: str, tier: str, stat) -> np.ndarray:
    """(n_w, 1, n_seed, N_TEST): stat(a) - stat(b) per matched patient, NaN if a cell is non-finite."""
    by = {(c["arch"], c["w"], c["seed"]): c for c in cells}
    out = np.full((len(w4.WEIGHTS), 1, len(s1.SEEDS_A), g.N_TEST), np.nan)
    for i, w in enumerate(w4.WEIGHTS):
        for k, sd in enumerate(s1.SEEDS_A):
            ca, cb = by[(a, w, sd)], by[(b, w, sd)]
            if s1.finite_a(ca) and s1.finite_a(cb):
                out[i, 0, k] = stat(ca[tier], cb[tier])
    return out


def ln_burden(ta, tb):
    f = lambda t: np.log(np.maximum(np.asarray(t["burden_integral"]), EPS))  # noqa: E731
    return f(ta) - f(tb)


def d_exposure(ta, tb):
    return np.asarray(ta["exposure"]) - np.asarray(tb["exposure"])


def wilcoxon_per_w(d: np.ndarray) -> dict:
    out = {}
    for i, w in enumerate(w4.WEIGHTS):
        x = d[i, 0][np.isfinite(d[i, 0])]
        out[str(w)] = float(wilcoxon(x, alternative="less").pvalue) if np.any(x != 0) else None
    return out


def one(name: str) -> dict:
    shard_dir, ref, mlp = TRAINERS[name]
    cells = load(shard_dir, (ref, mlp))
    res = {}
    for tier in s1.TIERS:
        lb = paired(cells, ref, mlp, tier, ln_burden)
        ex = paired(cells, ref, mlp, tier, d_exposure)
        res[tier] = {
            "ln_burden_ratio": w4.delta_ci(lb, (tier,)),
            "exposure": w4.delta_ci(ex, (tier,)),
            "wilcoxon_less_per_w": wilcoxon_per_w(lb),
            "arms": {a: v12.describe(cells, a, tier) for a in (ref, mlp)},
        }
        r = res[tier]
        r["burden_smaller"] = r["ln_burden_ratio"]["ci95"][1] < 0
        r["exposure_guard_ok"] = not (r["exposure"]["ci95"][0] > EXPOSURE_GUARD)
        r["passes"] = r["burden_smaller"] and r["exposure_guard_ok"]
    return res


def main() -> int:
    res = {"prereg": "adaptive_therapy/PREREG.md V15 [2026-10-22]", "trainers": {}}
    for name in TRAINERS:
        res["trainers"][name] = one(name)
    t = res["trainers"]
    same_sign_t1 = all(
        t[n]["tier1"]["ln_burden_ratio"]["delta"] < 0 for n in ("v13_r2d2", "v14_es")
    )
    res["stage1_passes"] = bool(
        all(t[n]["tier0"]["passes"] for n in ("v13_r2d2", "v14_es")) and same_sign_t1
    )
    (g.OUT / "v15_rescore.json").write_text(json.dumps(res, indent=2))
    for n, r in t.items():
        for tier in s1.TIERS:
            x = r[tier]
            print(
                f"{n:9} {tier}: ln B ratio {x['ln_burden_ratio']['delta']:+.3f} "
                f"{[round(v, 3) for v in x['ln_burden_ratio']['ci95']]}  exposure "
                f"{x['exposure']['delta']:+.1f} {[round(v, 1) for v in x['exposure']['ci95']]}  "
                f"passes={x['passes']}"
            )
    print("STAGE 1 passes:", res["stage1_passes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
