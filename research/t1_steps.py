"""T1: with 10x the gradient steps, does the full-information reference beat the MLP?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] T1, before any T1
cell ran. A follow-up program to S3 (RESEARCH_LOG L840), not a re-run of it:
S3's code runs unchanged with its trainer and output paths rebound below.

  Trainer   PQNConfig(batch_patients=1024, n_updates=3000); eps and lr
            schedules keep their fractions of training. All else as S3.
  Primary   S3's: Delta(O_full, MLP) at Tier 0 and R, same readings in order.

Run: JAX_PLATFORMS=cpu uv run python research/t1_steps.py
     -> research/outputs/t1_steps.{json,log}, t1_shards/, t1_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_budget as s3  # noqa: E402  (imports s2/s1/w4: backend env first)

from adaptive_therapy.rl import pqn  # noqa: E402

s1, s2, w4, g, wf = s3.s1, s3.s2, s3.w4, s3.g, s3.wf
s3.CFG = pqn.PQNConfig(batch_patients=1024, n_updates=3000)
s3.SHARDS = g.OUT / "t1_shards"
s3.POLICIES = g.OUT / "t1_policies"
S3_SHARDS = g.OUT / "s3_shards"
RESULT = g.OUT / "t1_steps.json"


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


def step_effect(cells: list[dict]) -> dict:
    """Tier-0 Delta U of each arm at 3000 updates vs its own S3 cell (300 updates)."""
    old = [
        {**json.loads(p.read_text()), "arch": f"{json.loads(p.read_text())['arch']}300"}
        for p in sorted(S3_SHARDS.glob("*.json"))
    ]
    allc = cells + old
    valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in allc}
    return {
        a: w4.comparison(allc, valid, a, f"{a}300", "level", s3.SEEDS, ("tier0",)) for a in s3.ARMS
    }


def main() -> int:
    s3.SHARDS.mkdir(parents=True, exist_ok=True)
    s2.reproduction_guard()
    jobs = [(a, w, s) for a in s3.ARMS for w in w4.WEIGHTS for s in s3.SEEDS]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    res = s3.analyse(cells)
    res["prereg"] = "adaptive_therapy/PREREG.md T1 [2026-09-30]"
    res["step_effect"] = step_effect(cells)
    for k in ("primary", "tier1"):
        print(f"O_full − MLP {k:8s} {s1.ci(res[k])}")
    for k, v in res["step_effect"].items():
        print(f"step effect {k:6s} (3000 − 300, Tier 0) {s1.ci(v)}")
    r = res["R"]
    print(
        f"R (train return, O_full − MLP) {r['mean']:+.2f}, higher in {r['cells_positive']}/{r['n']}"
    )
    print(f"READING: {res['reading']}")
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, w, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        (s3.SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(s3.cell(arch, w, seed)))
        raise SystemExit(0)
    raise SystemExit(main())
