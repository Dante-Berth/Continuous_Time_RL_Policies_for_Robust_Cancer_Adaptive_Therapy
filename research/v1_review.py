"""V1: three reviewer objections, answered from S1's saved policies only.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-10-02] V1 (2268849), before
any V1 number was computed. Post hoc on S1; no retraining.

  V1a  training return (own reward, own training cohort) vs U, per arm, S1 Part A
       (+ Part B descriptive). Delta R CI over seeds: hi < 0 => "loses the reward
       too"; lo > 0 => "misaligned"; else "ties the reward".
  V1b  median over (seed, patient) of the paired U difference, for every contrast
       the paper reports, beside S1's mean; same seed+patient bootstrap.
  V1c  S1 Part B policies + MTD, thr0.8 rolled out for 156 visits (1085 d);
       each learned arm - MTD at p 0.5.

Run: JAX_PLATFORMS=cpu uv run python research/v1_review.py
     -> research/outputs/v1_review.{json,log}, v1_shards/
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_diagnose as s2d  # noqa: E402  (imports s1_strobl: g.RANGES = "strobl2021")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402

s1 = s2d.s1
w4, w4b, g = s1.w4, s1.w4b, s1.g
SHARDS = g.OUT / "v1_shards"
RESULT = g.OUT / "v1_review.json"
LONG_VISITS = 156  # M1b's horizon, 1085 d
PART_ARCHS = {"a": s1.ARCHS_A, "b": tuple(w4.ARCHS)}
PART_SEEDS = {"a": s1.SEEDS_A, "b": s1.SEEDS_B}
LONG_RULES = ("MTD", "thr0.8")


def set_horizon(n_visits: int) -> None:
    """Rebind the episode length every imported helper reads (M1b's approach)."""
    g.EP = replace(g.EP, n_visits=n_visits)
    g.TS = np.arange(n_visits) * g.EP.visit_interval
    w4.HORIZON = float(g.TS[-1])


def load(part: str, arch: str, w: float, seed: int):
    like = w4.ARCHS[arch](jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(s1.POLICIES / part / f"{arch}_w{w}_s{seed}.eqx", like)


def worker_returns(part: str, arch: str) -> list[dict]:
    """V1a: return and U on the seed's training cohort (at the training drop rate)."""
    ep = replace(g.EP, drop_rate=0.0 if part == "a" else w4.TRAIN_DROP)
    out = []
    for s in PART_SEEDS[part]:
        tr = sample_cohort(g.FAM, g.N_TRAIN, seed=s, ranges=g.RANGES)
        th, y0 = jnp.asarray(tr.thetas), jnp.asarray(tr.y0s)
        for w in w4.WEIGHTS:
            sc = s2d.score(load(part, arch, w, s), ep, th, y0, w4.eval_key(s), w)
            out.append({"arch": arch, "w": w, "seed": s, **sc})
    return out


def worker_long(arch: str) -> list[dict]:
    """V1c: Part B policies (or a rule) on S1's Tier-0 test cohort over 1085 d."""
    set_horizon(LONG_VISITS)
    if arch in LONG_RULES:
        return [w4.rules_cell(arch, s) for s in s1.SEEDS_B]
    return [
        {"arch": arch, "w": w, "seed": s, **w4.evaluate(load("b", arch, w, s), s, w)}
        for w in w4.WEIGHTS
        for s in s1.SEEDS_B
    ]


def spawn(job: str, part: str, arch: str) -> list[dict]:
    """One process per job (RESEARCH_LOG L207: one process scoring hundreds of
    policies leaks JIT mappings until LLVM cannot allocate)."""
    shard = SHARDS / f"{job}_{part}_{arch}.json"
    if not shard.exists():  # resume: a finished shard is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", job, part, arch],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {job} {part} {arch}", flush=True)
    return json.loads(shard.read_text())


# --- V1a -------------------------------------------------------------------


def seed_ci(rows_x, rows_b, stat: str, seeds, seed: int = 0) -> dict:
    """Mean over (w, seed) of x - b, with a CI resampling seeds (2000 draws)."""
    bx = {(r["w"], r["seed"]): r[stat] for r in rows_x}
    bb = {(r["w"], r["seed"]): r[stat] for r in rows_b}
    d = np.array([[bx[(w, s)] - bb[(w, s)] for s in seeds] for w in w4.WEIGHTS])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), (w4.N_BOOT, len(seeds)))
    draws = np.array([d[:, i].mean() for i in idx])
    return {
        "delta": float(d.mean()),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "cells_positive": int((d > 0).sum()),
        "n_cells": int(d.size),
        "per_w": {str(w): float(v) for w, v in zip(w4.WEIGHTS, d.mean(1), strict=True)},
    }


def reading_a(ci) -> str:
    lo, hi = ci
    if hi < 0:
        return "loses the reward too: not reward misalignment"
    if lo > 0:
        return "misaligned: wins the reward, loses U"
    return "ties the reward"


def analyse_a(ret: dict) -> dict:
    out = {}
    for part, pairs in (
        ("a", (("lstm", "mlp"), ("ode_rnn", "mlp"), ("ode_rnn", "lstm"))),
        ("b", (("lstm", "mlp"), ("lstm_dt", "mlp"), ("ode_rnn", "mlp"), ("ode_rnn", "lstm"))),
    ):
        for x, b in pairs:
            r = {
                stat: seed_ci(ret[part][x], ret[part][b], stat, PART_SEEDS[part])
                for stat in ("return", "U")
            }
            if part == "a":
                r["reading"] = reading_a(r["return"]["ci95"])
            out[f"{part}|{x}-{b}"] = r
    return out


# --- V1b -------------------------------------------------------------------


def median_ci(d: np.ndarray, seed: int = 0) -> dict:
    """Mean over (w, p) of the median paired difference over (seed, patient).

    Same seed+patient resampling as w4.delta_draws (same rng draw order), with
    the median in place of the mean."""
    rng = np.random.default_rng(seed)
    n = d.shape[3]
    pi = rng.integers(0, n, (w4.N_BOOT, n))
    si = rng.integers(0, d.shape[2], (w4.N_BOOT, d.shape[2]))
    draws = np.array([np.nanmedian(d[:, :, s][..., i], axis=(2, 3)).mean() for s, i in zip(si, pi)])
    return {
        "delta": float(np.nanmedian(d, axis=(2, 3)).mean()),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
    }


def agreement(mean_ci, med_ci) -> str:
    def side(ci):
        return 1 if ci[0] > 0 else -1 if ci[1] < 0 else 0

    a, b = side(mean_ci), side(med_ci)
    if a == b:
        return "agrees"
    return "reversed" if a * b < 0 else "weaker"


def contrasts(cells, valid, pairs, seeds, drops) -> dict:
    out = {}
    for x, b in pairs:
        d = w4.paired(cells, valid, x, b, "level", seeds, drops)
        mean, med = w4.delta_ci(d, drops), median_ci(d)
        out[f"{x}-{b}"] = {
            "mean": {k: mean[k] for k in ("delta", "ci95")},
            "median": med,
            "agreement": agreement(mean["ci95"], med["ci95"]),
        }
    return out


def s1_cells(kind: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(s1.SHARDS.glob(f"{kind}_*.json"))]


def censoring(cells, archs, drops) -> dict:
    return {
        f"{a}|{p}": float(
            np.mean([np.mean(c[str(p)]["censored"]) for c in cells if c["arch"] == a])
        )
        for a in archs
        for p in drops
    }


def analyse_b() -> dict:
    a_cells = s1_cells("a")
    a_valid = {(c["arch"], c["w"], c["seed"]): s1.finite_a(c) for c in a_cells}
    pairs_a = (("ode_rnn", "mlp"), ("ode_rnn", "lstm"), ("lstm", "mlp"))
    res = {f"a|{t}": contrasts(a_cells, a_valid, pairs_a, s1.SEEDS_A, (t,)) for t in s1.TIERS}
    b_cells = w4b.with_mtd(s1_cells("b"), s1_cells("rb"))
    b_valid, _ = w4b.itt_validity(b_cells, (*w4.ARCHS, "MTD"), s1.SEEDS_B)
    pairs_b = tuple((x, "MTD") for x in w4.ARCHS) + (("ode_rnn", "mlp"), ("ode_rnn", "lstm"))
    res["b|0.5"] = contrasts(b_cells, b_valid, pairs_b, s1.SEEDS_B, (0.5,))
    # Reproduction guard: the mean must be S1's logged number, or nothing here is.
    logged = json.loads((g.OUT / "s1_strobl.json").read_text())
    for x in ("mlp", "lstm"):
        want = logged["A"]["H_ct_S"]["vs"][x]["tier1"]["delta"]
        got = res["a|tier1"][f"ode_rnn-{x}"]["mean"]["delta"]
        if abs(want - got) > 1e-6:
            raise RuntimeError(f"V1b does not reproduce S1 ODE-{x} Tier 1: {got} vs {want}")
    res["censoring_a"] = censoring(a_cells, s1.ARCHS_A, s1.TIERS)
    res["censoring_b"] = censoring(b_cells, (*w4.ARCHS, "MTD"), (0.0, 0.5))
    rev = [
        f"{k}|{c}"
        for k, v in res.items()
        if k.startswith(("a|", "b|"))
        for c, r in v.items()
        if r["agreement"] == "reversed"
    ]
    head = [
        res[f"a|{t}"][f"ode_rnn-{x}"]["median"]["ci95"][0] > 0
        for t in s1.TIERS
        for x in ("mlp", "lstm")
    ]
    head += [res["b|0.5"][f"{x}-MTD"]["median"]["ci95"][0] > 0 for x in w4.ARCHS]
    res["reversed"] = rev
    res["passes"] = not rev and not any(head)
    return res


# --- V1c -------------------------------------------------------------------


def reading_c(fb: dict) -> str:
    if any(v["ci95"][0] > 0 for v in fb.values()):
        return "a learned arm beats MTD over three years: the one-year statement is horizon-bound"
    if all(v["ci95"][1] < 0 for v in fb.values()):
        return "learned arms stay below MTD over three years"
    return "no arm above MTD"


def analyse_c(long_cells: list[dict], long_rules: list[dict]) -> dict:
    set_horizon(LONG_VISITS)
    cells = w4b.with_mtd(long_cells, long_rules)
    valid, arm_valid = w4b.itt_validity(cells, (*w4.ARCHS, "MTD"), s1.SEEDS_B)
    fb = {x: w4.comparison(cells, valid, x, "MTD", "level", s1.SEEDS_B, (0.5,)) for x in w4.ARCHS}
    fb = {x: {k: v[k] for k in ("delta", "ci95", "per_seed")} for x, v in fb.items()}
    return {
        "horizon_days": w4.HORIZON,
        "arm_valid": arm_valid,
        "vs_mtd_p0.5": fb,
        "reading": reading_c(fb),
        "median": contrasts(
            cells,
            valid,
            tuple((x, "MTD") for x in w4.ARCHS) + (("ode_rnn", "mlp"), ("ode_rnn", "lstm")),
            s1.SEEDS_B,
            (0.5,),
        ),
        "table": w4.table(cells, valid, long_rules, tuple(w4.ARCHS), s1.SEEDS_B),
    }


# --- report ----------------------------------------------------------------


def ci(v: dict) -> str:
    return f"{v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"


def report(res: dict) -> None:
    print("== V1a: training return vs U (X − Y, mean over 40 cells, seed CI) ==")
    for k, v in res["V1a"].items():
        r = v["return"]
        print(
            f"  {k:16s} ΔR {r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
            f" (>0 in {r['cells_positive']}/{r['n_cells']})  ΔU(train) {ci(v['U'])}"
            + (f"  -> {v['reading']}" if "reading" in v else "")
        )
    print("== V1b: mean vs median paired ΔU (seed+patient CI) ==")
    for k, v in res["V1b"].items():
        if not k.startswith(("a|", "b|")):
            continue
        for c, r in v.items():
            print(
                f"  {k:9s} {c:14s} mean {ci(r['mean']):28s} median {ci(r['median']):28s} {r['agreement']}"
            )
    print(f"  reversed {res['V1b']['reversed']}  PASS={res['V1b']['passes']}")
    print(f"  censored at 357 d: {json.dumps(res['V1b']['censoring_a'])}")
    print(f"  censored at 357 d (B): {json.dumps(res['V1b']['censoring_b'])}")
    c = res["V1c"]
    print(f"== V1c: {c['horizon_days']:.0f}-d replay, each arm − MTD at p 0.5 ==")
    for x, v in c["vs_mtd_p0.5"].items():
        print(f"  {x:8s} mean {ci(v):28s} median {ci(c['median'][f'{x}-MTD']['median'])}")
    for k in ("ode_rnn-mlp", "ode_rnn-lstm"):
        r = c["median"][k]
        print(f"  {k:14s} mean {ci(r['mean'])}  median {ci(r['median'])}")
    print(f"  reading: {c['reading']}  arm_valid {c['arm_valid']}")
    for name in (*w4.ARCHS, *LONG_RULES):
        t = c["table"].get(f"{name}|0.1|0.5")
        if t:
            print(
                f"  w0.1 p0.5 {name:8s} U {t['U']:6.1f}  TTP {t['ttp_median']:.0f} "
                f"[{t['ttp_iqr'][0]:.0f}–{t['ttp_iqr'][1]:.0f}]  cens {t['censoring']:.0%}  "
                f"exp {t['exposure_median']:.0f}  burden {t['burden_integral_median']:.3g}"
            )


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [("ret", p, a) for p, archs in PART_ARCHS.items() for a in archs]
    jobs += [("long", "b", a) for a in (*w4.ARCHS, *LONG_RULES)]
    with ThreadPoolExecutor(s1.wf.N_PARALLEL) as ex:
        out = dict(zip(jobs, ex.map(lambda j: spawn(*j), jobs), strict=True))
    ret = {p: {a: out[("ret", p, a)] for a in archs} for p, archs in PART_ARCHS.items()}
    res = {"prereg": "adaptive_therapy/PREREG.md V1 [2026-10-02]"}
    res["V1a"] = analyse_a(ret)
    res["V1b"] = analyse_b()
    long_cells = [c for a in w4.ARCHS for c in out[("long", "b", a)]]
    long_rules = [c for r in LONG_RULES for c in out[("long", "b", r)]]
    res["V1c"] = analyse_c(long_cells, long_rules)
    report(res)
    RESULT.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        job, part, arch = sys.argv[2:5]
        rows = worker_returns(part, arch) if job == "ret" else worker_long(arch)
        (SHARDS / f"{job}_{part}_{arch}.json").write_text(json.dumps(rows))
        raise SystemExit(0)
    raise SystemExit(main())
