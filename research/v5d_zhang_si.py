"""V5d: does one declared reading of Zhang 2017's unstated details reproduce its SI?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-02] V5d (92dffce), before
any V5d simulation ran. Targets from the Supplementary Information (3 pp.):
Supp. Table 1 (equilibrium mix per ordering) and Supp. Table 3 (times to 5% and
90% T-).

  Part 1  S1: exactly 22 orderings satisfy p.7's inequalities. S2: their
          equilibria match Supp. Table 1 (k = [1.5 TP, 1e4, 1e4]) within 2 pp.
  Part 2  readings R-a..R-d (ODE / discrete map x r-scale 0.1 / 0.01) on 40
          cells of Supp. Table 3; pass at >= 36.

  --v5e   PREREG V5e (7668890): the same Part 2, gated on S1 and the 22-row
          equilibrium match only. Decided after V5d exposed that the paper's
          Table 3 Patient #1 matrix is SI row 22, not row 7 (RESEARCH_LOG L1216).

Run: JAX_PLATFORMS=cpu uv run python research/v5d_zhang_si.py [--v5e]
     -> research/outputs/v5d_zhang_si.{json,log} | v5e_zhang_times.{json,log}
"""

from __future__ import annotations

import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_repro as v5  # noqa: E402  (enables x64)

from adaptive_therapy.envs import zhang2017 as z  # noqa: E402

RESULT = v5.OUT / "v5d_zhang_si.json"
VALUES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NEVER = None

# Supp. Table 1, second block (k = [1.5 TP, 10000, 10000]), % (T+, TP, T-), p.1.
SUPP1 = {
    9: (50, 50, 0), 10: (50, 50, 0), 15: (50, 50, 0), 16: (50, 50, 0),
    22: (44, 56, 0), 13: (47, 53, 0), 14: (47, 53, 0), 21: (47, 53, 0),
    7: (52, 48, 0), 8: (52, 48, 0), 20: (52, 48, 0), 19: (52, 48, 0),
    12: (48, 50, 2), 5: (45, 52, 3), 2: (41, 55, 3), 6: (38, 55, 7),
    18: (33, 47, 20), 1: (25, 54, 21), 4: (22, 51, 27),
    3: (0, 40, 60), 11: (0, 40, 60), 17: (0, 40, 60),
}  # fmt: skip
# Supp. Table 3, p.3: (MTD 5%, adaptive 5%, MTD 90%, adaptive 90%); None = "---".
SUPP3 = {
    12: (673, 673, 802, 4126), 5: (323, 323, 784, NEVER), 2: (161, 161, 759, 1888),
    6: (1, 1, 679, 1596), 18: (1, 1, 560, 560), 1: (1, 1, 532, 532),
    4: (1, 1, 501, 501), 3: (1, 1, 334, 334), 11: (1, 1, 334, 334), 17: (1, 1, 334, 334),
}  # fmt: skip
CELLS = ("MTD_5", "adaptive_5", "MTD_90", "adaptive_90")


def orderings() -> list[tuple]:
    """(a_p_tp, a_p_n, a_tp_p, a_tp_n, a_n_p, a_n_tp) under p.7's six inequalities.

    With 1 = T+, 2 = TP, 3 = T-: a31 > a21, a32 > a12, a13 > a23, a13 > a12,
    a23 > a21, a32 > a31.
    """
    out = []
    for a12, a13, a21, a23, a31, a32 in itertools.permutations(VALUES):
        if a31 > a21 and a32 > a12 and a13 > a23 and a13 > a12 and a23 > a21 and a32 > a31:
            out.append((a12, a13, a21, a23, a31, a32))
    return out


def theta(alphas, scale: float) -> np.ndarray:
    return np.array([r * scale for r in z.R_CELL_LINES] + list(alphas))


def mix(eq) -> np.ndarray:
    x = np.asarray(eq[:3], dtype=float)
    return 100.0 * x / x.sum()


def part1() -> dict:
    ords = orderings()
    eqs = [v5.zh_equilibrium(theta(a, 0.1)) for a in ords]
    ours = np.array([mix(e) for e in eqs])
    rows = list(SUPP1)
    theirs = np.array([SUPP1[r] for r in rows], dtype=float)
    cost = np.abs(ours[:, None, :] - theirs[None, :, :]).max(-1)
    oi, ri = linear_sum_assignment(cost)
    match = {rows[r]: int(o) for o, r in zip(oi, ri, strict=True)}
    worst = float(cost[oi, ri].max()) if len(ords) == len(rows) else float("inf")
    table3 = {
        name: rows[
            int(np.argmin(np.abs(mix(v5.zh_equilibrium(theta(a, 0.1)))[None] - theirs).max(-1)))
        ]
        for name, a in (("patient1", z.PATIENT_1), ("patient2", z.PATIENT_2))
    }
    return {
        "n_orderings": len(ords),
        "S1": len(ords) == 22,
        "worst_pp": worst,
        "S2_multiset": worst <= 2.0,
        "table3_rows": table3,
        "S2_table3": table3 == {"patient1": 7, "patient2": 5},
        "per_row": {
            r: {"ours": ours[o].round(1).tolist(), "theirs": SUPP1[r], "alphas": ords[o]}
            for r, o in match.items()
        },
        "_match": match,
        "_ords": ords,
    }


def map_run(th: np.ndarray, regimen: str, days: int = 10_000) -> dict:
    """R-c / R-d: the discrete-time map, one step per unit (p.8's 'each time step')."""
    import jax.numpy as jnp

    eq = v5.zh_equilibrium(jnp.asarray(th))
    psa_eq = eq[3]
    y = np.r_[0.25 * eq[:3], 0.25 * psa_eq]
    start, base, drug, frac = None, None, 0.0, []
    for day in range(days):
        if start is None and y[3] >= 0.8 * psa_eq:
            start, base, drug = day, y[3], 1.0
        elif start is not None and regimen == "adaptive":
            drug = 0.0 if y[3] < 0.5 * base else 1.0 if y[3] >= base else drug
        frac.append(y[2] / max(y[:3].sum(), 1e-10))
        y = np.maximum(y + np.asarray(z.vector_field(0.0, jnp.asarray(y), (th, drug))), 0.0)
    frac = np.array(frac)

    def first(t):
        i = np.flatnonzero(frac >= t)
        return int(i[0]) if i.size else NEVER

    return {"start": start, "t_5pct": first(0.05), "t_90pct": first(0.90)}


READINGS = {"R-a": ("ode", 0.1), "R-b": ("ode", 0.01), "R-c": ("map", 0.1), "R-d": ("map", 0.01)}


def run_row(reading: str, alphas) -> dict:
    kind, scale = READINGS[reading]
    th = theta(alphas, scale)
    out = {}
    for reg in ("MTD", "adaptive"):
        r = v5.zh_run(th, reg) if kind == "ode" else map_run(th, reg)
        out[f"{reg}_5"], out[f"{reg}_90"] = r["t_5pct"], r["t_90pct"]
    return out


def hit(got, want) -> bool:
    if want is NEVER or got is NEVER:
        return want is got
    return abs(got - want) <= max(0.03 * want, 3)


def part2(p1: dict) -> dict:
    jobs = [(rd, row) for rd in READINGS for row in SUPP3]
    alphas = {row: p1["_ords"][p1["_match"][row]] for row in SUPP3}
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda j: run_row(j[0], alphas[j[1]]), jobs))
    res = {rd: {"rows": {}, "hits": 0} for rd in READINGS}
    for (rd, row), g in zip(jobs, got, strict=True):
        want = dict(zip(CELLS, SUPP3[row], strict=True))
        cells = {c: [g[c], want[c], hit(g[c], want[c])] for c in CELLS}
        res[rd]["rows"][row] = cells
        res[rd]["hits"] += sum(v[2] for v in cells.values())
    for rd in READINGS:
        res[rd]["pass"] = res[rd]["hits"] >= 36
    passing = [rd for rd in READINGS if res[rd]["pass"]]
    res["reading"] = (
        f"reproduces under {passing[0]}"
        if len(passing) == 1
        else "not reproducible from paper + SI"
        if not passing
        else f"ambiguous: {passing}"
    )
    return res


def main(v5e: bool = False) -> int:
    p1 = part1()
    print(
        f"== Part 1 == orderings {p1['n_orderings']} (S1 {p1['S1']}); worst match {p1['worst_pp']:.1f} pp (S2 {p1['S2_multiset']})"
    )
    print(
        f"  Table 3 matrices land on rows {p1['table3_rows']} (want patient1 7, patient2 5): {p1['S2_table3']}"
    )
    for r, v in sorted(p1["per_row"].items()):
        print(f"  row {r:2d} ours {v['ours']}  theirs {list(v['theirs'])}  a={v['alphas']}")
    tag = "V5e" if v5e else "V5d"
    res = {"prereg": f"adaptive_therapy/PREREG.md {tag} [2026-10-02]"}
    res["part1"] = {k: v for k, v in p1.items() if not k.startswith("_")}
    if p1["S1"] and p1["S2_multiset"] and (v5e or p1["S2_table3"]):
        p2 = part2(p1)
        res["part2"] = p2
        print("== Part 2 ==")
        for rd in READINGS:
            print(f"  {rd}: {p2[rd]['hits']}/40  pass={p2[rd]['pass']}")
            for row, cells in p2[rd]["rows"].items():
                print(
                    f"     row {row:2d} "
                    + "  ".join(f"{c} {v[0]}/{v[1]}{'' if v[2] else '✗'}" for c, v in cells.items())
                )
        print(f"READING: {p2['reading']}")
    else:
        print("Part 1 failed: Part 2 not run, as registered")
    out = v5.OUT / "v5e_zhang_times.json" if v5e else RESULT
    out.write_text(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(v5e="--v5e" in sys.argv))
