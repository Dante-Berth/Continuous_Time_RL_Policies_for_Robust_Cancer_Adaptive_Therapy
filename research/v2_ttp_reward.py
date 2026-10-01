"""V2: with a reward whose survival term stops at progression, does the
full-information reference beat the burden-only MLP?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-02] V2, before any V2
cell ran. T1's design (RESEARCH_LOG L1042) with one change: the reward is
RewardConfig(survival_until="progression"), so its survival total is the TTP
plus the 28-d confirmation window. T1's code runs unchanged with the reward and
output paths rebound below.

  Primary   T1's: Delta(O_full, MLP) at Tier 0 and R, same readings in order.

Run: JAX_PLATFORMS=cpu uv run python research/v2_ttp_reward.py
     -> research/outputs/v2_ttp_reward.{json,log}, v2_shards/, v2_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_steps as t1  # noqa: E402  (rebinds S3 to T1's trainer: backend env first)

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402

s1, s2, s3, w4, g, wf = t1.s1, t1.s2, t1.s3, t1.w4, t1.g, t1.wf
s2d = t1.s3.s2d
# The one change. Every helper that builds a reward looks the name up at call
# time: w4.train_policy (training), w4.evaluate (U ignores the reward) and
# s2d.score (R, the training return).
V2_REWARD = partial(RewardConfig, survival_until="progression")
w4.RewardConfig = V2_REWARD
s2d.RewardConfig = V2_REWARD
s3.SHARDS = g.OUT / "v2_shards"
s3.POLICIES = g.OUT / "v2_policies"
T1_SHARDS = g.OUT / "t1_shards"
RESULT = g.OUT / "v2_ttp_reward.json"


def spawn(arch: str, w: float, seed: int) -> dict:
    shard = s3.SHARDS / f"{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def reward_effect(cells: list[dict]) -> dict:
    """Tier-0 Delta U of each arm under V2's reward vs its own T1 cell."""
    old = [
        {**json.loads(p.read_text()), "arch": f"{json.loads(p.read_text())['arch']}_t1"}
        for p in sorted(T1_SHARDS.glob("*.json"))
    ]
    allc = cells + old
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in allc}
    return {
        a: w4.comparison(allc, valid, a, f"{a}_t1", "level", s3.SEEDS, ("tier0",)) for a in s3.ARMS
    }


def main() -> int:
    s3.SHARDS.mkdir(parents=True, exist_ok=True)
    s2.reproduction_guard()
    jobs = [(a, w, s) for a in s3.ARMS for w in w4.WEIGHTS for s in s3.SEEDS]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    res = s3.analyse(cells)
    res["prereg"] = "adaptive_therapy/PREREG.md V2 [2026-10-02]"
    res["reward"] = V2_REWARD().to_dict()
    res["reward_effect"] = reward_effect(cells)
    for k in ("primary", "tier1"):
        print(f"O_full − MLP {k:8s} {s1.ci(res[k])}")
    for k, v in res["reward_effect"].items():
        print(f"reward effect {k:6s} (V2 − T1, Tier 0) {s1.ci(v)}")
    r = res["R"]
    print(
        f"R (train return, O_full − MLP) {r['mean']:+.2f}, higher in {r['cells_positive']}/{r['n']}"
    )
    for name in ("mlp", "o_full", "thr0.8", "MTD"):
        t = res["table"].get(f"{name}|0.05|tier0")
        if t:
            print(
                f"  w0.05 tier0 {name:7s} U {t['U']:6.1f}  TTP {t['ttp_median']:.0f} "
                f"[{t['ttp_iqr'][0]:.0f}–{t['ttp_iqr'][1]:.0f}]  exp {t['exposure_median']:.0f}  "
                f"burden {t['burden_integral_median']:.3g}"
            )
    print(f"READING: {res['reading']}")
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        if w4.RewardConfig().survival_until != "progression":
            raise RuntimeError("V2's reward was not rebound; refusing to train T1 again")
        arch, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        (s3.SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(s3.cell(arch, w, seed)))
        raise SystemExit(0)
    raise SystemExit(main())
