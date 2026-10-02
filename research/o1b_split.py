"""O1b: is O1's headroom full information, or one fixed threshold the MLP did not learn?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-02] O1b (11be513), after
O1's descriptive output and before this split was computed.

  global  per w, the ThresholdPolicy level maximising mean U on the other nine
          seeds' cohorts (leave one seed out), scored on the held-out seed
  Delta_G = U(global) - U(MLP)    learnable without memory
  Delta_I = U(tuned)  - U(global) value of patient-specific information

Run: JAX_PLATFORMS=cpu uv run python research/o1b_split.py
     -> research/outputs/o1b_split.{json,log}, o1b_shards/
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o1_oracle as o1  # noqa: E402  (imports s1_strobl: backend env first)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.policies.rules import ThresholdPolicy  # noqa: E402
from adaptive_therapy.rl.rollout import batch_rollout  # noqa: E402

s1, w4, g = o1.s1, o1.w4, o1.g
SEEDS, LEVELS = o1.SEEDS, o1.LEVELS
SHARDS = g.OUT / "o1b_shards"
RESULT = g.OUT / "o1b_split.json"


def worker(seed: int) -> dict:
    """Per level: noise-free endpoints (O1's rollout) and noisy ones (S1's)."""
    th, y0 = (jnp.asarray(x) for x in w4.tier0_cohort(g.N_TEST, seed))
    out = {"seed": seed, "clean": [], "noisy": []}
    for x in LEVELS:
        p = ThresholdPolicy(float(x))
        tr = batch_rollout(p, th, y0, o1.KEY, o1.EP, o1.RewardConfig(), greedy=True)
        out["clean"].append(o1.endpoints(tr["burden"], tr["dose"]))
        tr = batch_rollout(p, th, y0, w4.eval_key(seed), g.EP, o1.RewardConfig(), greedy=True)
        out["noisy"].append(o1.endpoints(tr["burden"], tr["dose"]))
    return out


def spawn(seed: int) -> dict:
    shard = SHARDS / f"s{seed}.json"
    if not shard.exists():
        subprocess.run(
            [sys.executable, __file__, "--worker", str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    return json.loads(shard.read_text())


def reading(dg, di) -> str:
    if di["ci95"][1] < o1.HEADROOM_D and dg["ci95"][0] > o1.HEADROOM_D:
        return "the headroom is a fixed threshold PQN's reward did not reach, not information"
    if di["ci95"][0] > o1.HEADROOM_D:
        return "patient-specific information is worth > 10 d: memory has something to recover"
    return "mixed"


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    lv = {s: spawn(s) for s in SEEDS}
    o1s = {s: json.loads((o1.SHARDS / f"s{s}.json").read_text()) for s in SEEDS}
    cells, chosen = [], {}
    for w in w4.WEIGHTS:
        mean_u = {s: np.array([o1.u_of(e, w).mean() for e in lv[s]["clean"]]) for s in SEEDS}
        for s in SEEDS:
            j = int(np.argmax(sum(mean_u[t] for t in SEEDS if t != s)))
            chosen[f"{w}|{s}"] = float(LEVELS[j])
            cells.append({"arch": "global", "w": w, "seed": s, "tier0": lv[s]["clean"][j]})
            cells.append({"arch": "global_noisy", "w": w, "seed": s, "tier0": lv[s]["noisy"][j]})
            cells.append(
                {"arch": "tuned", "w": w, "seed": s, "tier0": o1s[s]["w"][str(w)]["tuned"]}
            )
            m = json.loads((s1.SHARDS / f"a_mlp_w{w}_s{s}.json").read_text())["tier0"]
            cells.append({"arch": "mlp", "w": w, "seed": s, "tier0": m})
    valid = {(c["arch"], c["w"], c["seed"]): True for c in cells}

    def cmp(a, b):
        return w4.comparison(cells, valid, a, b, "level", SEEDS, ("tier0",))

    res = {
        "prereg": "adaptive_therapy/PREREG.md O1b [2026-10-02]",
        "chosen_levels": chosen,
        "delta_G": cmp("global", "mlp"),
        "delta_I": cmp("tuned", "global"),
        "global_noisy_minus_mlp": cmp("global_noisy", "mlp"),
        "table": w4.table(
            cells, valid, [], ("global", "global_noisy", "tuned", "mlp"), SEEDS, ("tier0",)
        ),
    }
    res["reading"] = reading(res["delta_G"], res["delta_I"])
    print(
        f"chosen levels by w: { {w: sorted({v for k, v in chosen.items() if k.startswith(f'{w}|')}) for w in w4.WEIGHTS} }"
    )
    for k in ("delta_G", "delta_I", "global_noisy_minus_mlp"):
        print(
            f"{k:24s} {s1.ci(res[k])}   per w {json.dumps({kk: round(v, 1) for kk, v in res[k]['per_w_p'].items()})}"
        )
    for name in ("global", "global_noisy", "tuned", "mlp"):
        for w in (0.05, 0.4):
            t = res["table"][f"{name}|{w}|tier0"]
            print(
                f"  w{w} {name:12s} U {t['U']:6.1f}  TTP {t['ttp_median']:.0f} [{t['ttp_iqr'][0]:.0f}–{t['ttp_iqr'][1]:.0f}]"
                f"  cens {t['censoring']:.0%}  exp {t['exposure_median']:.0f}  burden {t['burden_integral_median']:.3g}"
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
