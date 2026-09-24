"""W3c: memory vs continuous time, per-cell validity, the trained objective as primary.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-24] W3c, before any W3c
cell ran. Cells are trained by research/w3_frontier.py exactly as in W3b, on
fresh seeds {6..10}; this script adds the W3c analysis:

  Validity  a cell with Tier-1 closed-loop < 0.10 is EXCLUDED; an arm is INVALID
            if any weight has < 3 of 5 valid seeds or > 20% of its cells are
            excluded. Decided once on the full data, before any bootstrap.
  Primary   U = min(TTP, 357) - w * cumulative exposure per patient (the
            survival and exposure terms of envs/reward.py). Delta(A, B) = mean
            over the 4 weights of the mean paired U_A - U_B over patient x seed
            pairs where both cells are valid. 2000 patient-bootstrap resamples.
            H passes iff the Tier-1 CI lies wholly above 0.
  Rules     none / MTD / AT50 / thr0.8 on the same cohorts and eval keys,
            descriptive only.
  Secondary W3b's RMST frontier gap on the valid cells, and per-w one-sided
            paired Wilcoxon on U.

Run: JAX_PLATFORMS=cpu uv run python research/w3c_utility.py w3c
     -> research/outputs/w3c_utility.json, w3c_utility.png (+ w3c_shards/)
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_frontier as wf  # noqa: E402  (REG is read from argv: pass "w3c")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.eval.metrics import (  # noqa: E402
    cumulative_exposure,
    paired_wilcoxon,
    ttp_running_nadir,
)
from adaptive_therapy.policies.rules import RULES, RulePolicy  # noqa: E402
from adaptive_therapy.rl import ppo  # noqa: E402

g = wf.g
GATE = g.CLOSED_LOOP_MIN
MIN_VALID_SEEDS = 3
MAX_EXCLUDED = 0.20
HYPOTHESES = {"H_mem": ("lstm", "mlp"), "H_ct": ("ode_rnn", "lstm")}


def utility(cell_tier: dict, w: float) -> np.ndarray:
    return np.asarray(cell_tier["ttp_capped"]) - w * np.asarray(cell_tier["exposure"])


def validity(cells: list[dict]) -> tuple[dict, dict]:
    """(valid[(arch, w, seed)] -> bool, arm_valid[arch] -> bool)."""
    valid = {(c["arch"], c["w"], c["seed"]): c["tier1"]["closed_loop"] >= GATE for c in cells}
    arm_valid = {}
    for a in wf.ARCHS:
        per_w = [sum(valid[(a, w, s)] for s in wf.SEEDS) for w in wf.WEIGHTS]
        excluded = sum(not valid[(a, w, s)] for w in wf.WEIGHTS for s in wf.SEEDS)
        n = len(wf.WEIGHTS) * len(wf.SEEDS)
        arm_valid[a] = min(per_w) >= MIN_VALID_SEEDS and excluded / n <= MAX_EXCLUDED
    return valid, arm_valid


def delta(cells, valid, a, b, tier, idx) -> tuple[float, dict]:
    """Mean over w of the mean paired U_A - U_B on patient indices `idx`."""
    by = {(c["arch"], c["w"], c["seed"]): c for c in cells}
    per_w = {}
    for w in wf.WEIGHTS:
        diffs = [
            utility(by[(a, w, s)][tier], w)[idx] - utility(by[(b, w, s)][tier], w)[idx]
            for s in wf.SEEDS
            if valid[(a, w, s)] and valid[(b, w, s)]
        ]
        per_w[w] = float(np.mean(np.concatenate(diffs))) if diffs else float("nan")
    return float(np.nanmean(list(per_w.values()))), per_w


def compare(cells, valid, a, b, tier, seed=0) -> dict:
    full = np.arange(g.N_TEST)
    est, per_w = delta(cells, valid, a, b, tier, full)
    rng = np.random.default_rng(seed)
    boots = [
        delta(cells, valid, a, b, tier, rng.integers(0, g.N_TEST, g.N_TEST))[0]
        for _ in range(wf.N_BOOT)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    by = {(c["arch"], c["w"], c["seed"]): c for c in cells}
    wilcoxon = {}
    for w in wf.WEIGHTS:
        pairs = [
            (s, by[(a, w, s)], by[(b, w, s)])
            for s in wf.SEEDS
            if valid[(a, w, s)] and valid[(b, w, s)]
        ]
        if pairs:
            ua = np.concatenate([utility(x[tier], w) for _, x, _ in pairs])
            ub = np.concatenate([utility(y[tier], w) for _, _, y in pairs])
            wilcoxon[str(w)] = paired_wilcoxon(ua, ub, alternative="greater")["p_value"]
    return {
        "delta": est,
        "ci95": [float(lo), float(hi)],
        "per_w": per_w,
        "wilcoxon_greater_p": wilcoxon,
    }


def rules_eval() -> dict:
    """Each rule on each seed's cohorts, with the cells' own eval key per seed."""
    out = {}
    for kind in RULES:
        for s in wf.SEEDS:
            k_eval = jax.random.split(jax.random.PRNGKey(s), 3)[2]
            for tier, cohort in wf.TIERS.items():
                th, y0 = cohort(g.N_TEST, s)
                ev = ppo.evaluate(
                    RulePolicy(kind), jnp.asarray(th), jnp.asarray(y0), k_eval, g.EP, RewardConfig()
                )
                burden, dose = np.asarray(ev["burden"]), np.asarray(ev["dose"])
                out[(kind, s, tier)] = {
                    "ttp_capped": ttp_running_nadir(burden, g.TS).days,
                    "exposure": cumulative_exposure(dose, g.TS),
                }
    return out


def summary_table(cells, valid, rules) -> dict:
    """Mean U, RMST and median exposure per (policy, w, tier), valid cells only."""
    tab = {}
    for tier in wf.TIERS:
        for w in wf.WEIGHTS:
            for a in wf.ARCHS:
                cs = [
                    c[tier]
                    for c in cells
                    if c["arch"] == a and c["w"] == w and valid[(a, w, c["seed"])]
                ]
                if cs:
                    tab[f"{a}|{w}|{tier}"] = {
                        "U": float(np.mean([utility(c, w).mean() for c in cs])),
                        "rmst": float(np.mean([np.mean(c["ttp_capped"]) for c in cs])),
                        "exposure": float(np.median([np.median(c["exposure"]) for c in cs])),
                        "n_seeds": len(cs),
                    }
            for kind in RULES:
                rs = [rules[(kind, s, tier)] for s in wf.SEEDS]
                tab[f"{kind}|{w}|{tier}"] = {
                    "U": float(np.mean([(r["ttp_capped"] - w * r["exposure"]).mean() for r in rs])),
                    "rmst": float(np.mean([r["ttp_capped"].mean() for r in rs])),
                    "exposure": float(np.median([np.median(r["exposure"]) for r in rs])),
                    "n_seeds": len(rs),
                }
    return tab


def plot(tab, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "mlp": "#1F3B4D",
        "lstm": "#8E7DBE",
        "ode_rnn": "#2A9D8F",
        "none": "#9A9A9A",
        "MTD": "#D1495B",
        "AT50": "#E09F3E",
        "thr0.8": "#3C6E71",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tier in zip(axes, wf.TIERS, strict=True):
        for name, col in colors.items():
            ys = [tab.get(f"{name}|{w}|{tier}", {}).get("U", np.nan) for w in wf.WEIGHTS]
            ls = "--" if name in RULES else "-"
            ax.plot(wf.WEIGHTS, ys, marker="o", ls=ls, color=col, label=name)
        ax.set_title(tier.replace("tier", "Tier "))
        ax.set_xlabel("exposure weight w")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("U = RMST − w · exposure (days)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyse(cells: list[dict]) -> dict:
    valid, arm_valid = validity(cells)
    res = {
        "prereg": "adaptive_therapy/PREREG.md W3c [2026-09-24]",
        "excluded": [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok],
        "arm_valid": arm_valid,
    }
    for h, (a, b) in HYPOTHESES.items():
        r = {t: compare(cells, valid, a, b, t) for t in wf.TIERS}
        ok = arm_valid[a] and arm_valid[b]
        r["valid"] = bool(ok)
        r["passes"] = bool(ok and r["tier1"]["ci95"][0] > 0)
        r["robust_wording_allowed"] = bool(
            r["passes"] and r["tier1"]["ci95"][0] > r["tier0"]["delta"]
        )
        by_arm = {
            x: [c for c in cells if c["arch"] == x and valid[(x, c["w"], c["seed"])]]
            for x in (a, b)
        }
        r["secondary_frontier"] = {t: wf.compare(by_arm, a, b, t) for t in wf.TIERS}
        res[h] = r
    return res, valid


def main() -> int:
    assert wf.REG == "w3c", "run as: research/w3c_utility.py w3c"
    wf.SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, s) for a in wf.ARCHS for w in wf.WEIGHTS for s in wf.SEEDS]
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: wf.spawn(*j), jobs))
    res, valid = analyse(cells)
    rules = rules_eval()
    res["table"] = summary_table(cells, valid, rules)
    for h in HYPOTHESES:
        r = res[h]
        print(
            f"{h} {HYPOTHESES[h]}: "
            + "  ".join(
                f"{t}: Δ {r[t]['delta']:+.1f} d [{r[t]['ci95'][0]:+.1f}, {r[t]['ci95'][1]:+.1f}]"
                for t in wf.TIERS
            )
            + f"  valid={r['valid']} PASS={r['passes']} robust={r['robust_wording_allowed']}"
        )
    print("arm valid:", res["arm_valid"], " excluded:", res["excluded"])
    for k, v in res["table"].items():
        if k.endswith("tier1"):
            print(
                f"  {k:24s} U {v['U']:7.1f}  RMST {v['rmst']:6.1f}  expo {v['exposure']:6.1f}  n={v['n_seeds']}"
            )
    (g.OUT / "w3c_utility.json").write_text(json.dumps(res, indent=2))
    plot(res["table"], g.OUT / "w3c_utility.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
