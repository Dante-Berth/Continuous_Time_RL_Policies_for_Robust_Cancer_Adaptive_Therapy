"""M1b: M1 with a three-year horizon (156 weekly visits, 1085 d).

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-30] M1b, before any M1b
evaluation ran. The one change from M1 is the horizon: M1's code runs unchanged
with its episode settings rebound below. The cohort seeds are M1's, so these
are the same patients followed three times longer.

  H_cost_long  at p 0.5 under hold, highest-cost Strobl bin - no-cost-end bin:
               CI < 0 in both turnover bins => not a horizon artefact; any CI
               > 0, or both covering 0 => at least partly the horizon; else mixed.

Run: JAX_PLATFORMS=cpu uv run python research/m1b_horizon.py
     -> research/outputs/m1b_horizon.{json,log}, m1b_shards/
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_monitoring as m1  # noqa: E402  (imports w4_irregular: backend env first)

import json  # noqa: E402

import numpy as np  # noqa: E402

from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402

g = m1.g
N_VISITS = 156
m1.EP = replace(g.EP, n_visits=N_VISITS)
m1.TS = np.arange(N_VISITS) * g.EP.visit_interval
m1.HORIZON = float(m1.TS[-1])
m1.SHARDS = g.OUT / "m1b_shards"
m1.SCRIPT = Path(__file__)
m1.RESULT = g.OUT / "m1b_horizon.json"


def indefinite_control_share() -> dict:
    """Test-cohort share meeting Strobl's criterion d_T > (1 - 1.2 n0) r_R (p.9).

    Context only: the criterion is derived for Strobl's baseline progression
    rule, not our running-nadir rule.
    """
    out = {}
    for b, (rr, dt) in enumerate(m1.BINS):
        shares = []
        for seed in m1.SEEDS:
            c = sample_cohort(
                g.FAM,
                m1.N,
                seed=2 * 100_000 + 1000 * b + seed,  # M1's test-cohort seed
                ranges="strobl2021",
                bounds={"r_R/r_S": rr, "d_T/r_S": dt},
            )
            th = c.thetas
            n0 = c.y0s.sum(1) / th[:, 2]
            shares.append(np.mean(th[:, 3] / th[:, 0] > (1 - 1.2 * n0) * th[:, 1] / th[:, 0]))
        out[m1.label(b)] = float(np.mean(shares))
    return out


def reading(cis) -> str:
    if all(hi < 0 for _, hi in cis):
        return "not a horizon artefact"
    if any(lo > 0 for lo, _ in cis) or all(lo <= 0 <= hi for lo, hi in cis):
        return "at least partly the horizon"
    return "mixed"


def main() -> int:
    m1.SHARDS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(m1.wf.N_PARALLEL) as ex:
        list(ex.map(m1.spawn, m1.SEEDS))
    res = m1.analyse()
    res["prereg"] = "adaptive_therapy/PREREG.md M1b [2026-09-30]"
    res["horizon_days"] = m1.HORIZON
    res["H_cost_long_reading"] = reading([v["ci95"] for v in res["H_cost"].values()])
    res["indefinite_control_share"] = indefinite_control_share()
    m1.report(res)
    print(f"H_cost_long: {res['H_cost_long_reading']}")
    for name, b in res["bins"].items():
        mtd = b["p0.5_hold_standing"]["MTD"]
        share = res["indefinite_control_share"][name]
        print(
            f"  {name:28s} MTD censored {mtd['censoring']:.0%} at {m1.HORIZON:.0f} d; "
            f"Strobl indefinite control {share:.0%}"
        )
    m1.RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        m1.worker(int(sys.argv[2]))
        raise SystemExit(0)
    raise SystemExit(main())
