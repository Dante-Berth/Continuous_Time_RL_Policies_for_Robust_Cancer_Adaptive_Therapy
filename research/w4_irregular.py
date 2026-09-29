"""W4: irregular observation -- does continuous time help when visits are missed?

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-25] W4, before any W4
cell ran. A missed visit gives no observation and no decision; the last chosen
dose is given again (rl/rollout.py). TTP and exposure are scored on the weekly
ground-truth grid at every drop rate.

  Train     {MLP, LSTM, LSTM+dt, ODE-RNN} x w {.05, .10, .20, .40} x seeds
            {11..20}, PQN (Phase-2c settings), Tier-0 training cohorts with 35%
            of visits missed at random. 160 cells, each in a fresh process.
  Test      the Tier-0 test patients at drop rates {0, .2, .5}: nested visit
            masks from each seed's eval key, the same for every arm and rule.
  Validity  per cell, greedy closed-loop >= 0.10 at drop 0; an arm is INVALID
            if any w has < 60% valid seeds or > 20% of its cells are
            excluded (W3c's rule, 3 of 5). Decided once, before any bootstrap.
  H_ct4     (primary) U = min(TTP, 357) - w * exposure per patient. Delta(ODE,
            X) = mean over w and p in {.2, .5} of the mean paired U_ODE - U_X
            over valid patient x seed pairs; 2000 bootstrap resamples of seeds
            AND patients. Passes iff the 95% CI lies above 0 for EVERY X in
            {MLP, LSTM, LSTM+dt} (intersection-union: no multiplicity correction).
  H_grace   (secondary) the same rule on the degradation U(p) - U(0).
  H_zs      (secondary) zero-shot: W3c's regular-trained policies (seeds 6..10)
            on the same test, X in {MLP, LSTM}. LSTM+dt has no regular-trained
            version: with dt = 7 throughout training its dt weight is a bias.
  Rules     none / MTD / AT50 / thr0.8 at each drop rate, descriptive only.

Run: JAX_PLATFORMS=cpu uv run python research/w4_irregular.py
     -> research/outputs/w4_irregular.{json,png}, w4_shards/, w4_policies/
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_frontier as wf  # noqa: E402  (imports phase2_gate: backend env first)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from w1_diagnose import tier0_cohort  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import (  # noqa: E402
    burden_integral,
    cumulative_exposure,
    paired_wilcoxon,
    ttp_running_nadir,
)
from adaptive_therapy.policies.lstm import LSTMPolicy  # noqa: E402
from adaptive_therapy.policies.rules import RULES, RulePolicy  # noqa: E402
from adaptive_therapy.rl import ppo, pqn  # noqa: E402
from adaptive_therapy.rl.rollout import observation_dependence  # noqa: E402

g = wf.g
ARCHS = {**wf.ARCHS, "lstm_dt": lambda k: LSTMPolicy(k, layer_norm=True, dt_input=True)}
WEIGHTS = (0.05, 0.10, 0.20, 0.40)
SEEDS = tuple(range(11, 21))  # 10: training variance is the unit (delta_ci)
ZS_ARCHS = ("mlp", "lstm", "ode_rnn")
ZS_SEEDS = (6, 7, 8, 9, 10)  # W3c's; its policies are in outputs/w3c_policies
TRAIN_DROP = 0.35
DROPS = (0.0, 0.2, 0.5)
TEST_DROPS = (0.2, 0.5)  # the Deltas average over these; drop 0 is the reference
HORIZON = float(g.TS[-1])
MIN_VALID_FRACTION = 0.6  # W3c's 3 of 5
MAX_EXCLUDED = 0.20
N_BOOT = 2000
HYPOTHESES = {
    "H_ct4": ("train", "level", ("mlp", "lstm", "lstm_dt")),
    "H_grace": ("train", "degradation", ("mlp", "lstm", "lstm_dt")),
    "H_zs": ("zs", "level", ("mlp", "lstm")),
}
SHARDS = g.OUT / "w4_shards"


def eval_key(seed: int):
    """The W3c cell eval key: zero-shot drop 0 must reproduce W3c's Tier 0."""
    return jax.random.split(jax.random.PRNGKey(seed), 3)[2]


def evaluate(policy, seed: int, w: float, cohort=tier0_cohort, drops=DROPS, ep=None) -> dict:
    """A test cohort (Tier 0 by default) at every drop rate, one eval key for all.

    `ep` replaces g.EP as the base episode (S2's reference sets its oracle)."""
    base = ep or g.EP
    th, y0 = (jnp.asarray(x) for x in cohort(g.N_TEST, seed))
    rw, k = RewardConfig(exposure_penalty=w), eval_key(seed)
    out = {}
    for p in drops:
        ev = ppo.evaluate(policy, th, y0, k, replace(base, drop_rate=p), rw)
        burden = np.asarray(ev["burden"])
        ttp = ttp_running_nadir(burden, g.TS)
        out[str(p)] = {
            "ttp_capped": ttp.days.tolist(),
            "censored": ttp.censored.tolist(),
            "exposure": cumulative_exposure(np.asarray(ev["dose"]), g.TS).tolist(),
            "burden_integral": burden_integral(burden, g.TS).tolist(),
            "seen_fraction": float(np.asarray(ev["seen"]).mean()),
        }
        if p == 0.0:
            out["closed_loop"] = observation_dependence(policy, ev["obs"], base, k, greedy=True)
    return out


def train_policy(
    arch: str,
    w: float,
    seed: int,
    train_drop: float = TRAIN_DROP,
    ep=None,
    make=None,
    cfg=None,
    n_train=None,
):
    """PQN on the seed's training cohort; returns (policy, train seconds).

    `make` overrides ARCHS[arch] and `ep` the base episode (S2's reference);
    `cfg` and `n_train` the trainer and cohort size (S3's budget)."""
    k_init, k_train, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
    c = sample_cohort(g.FAM, n_train or g.N_TRAIN, seed=seed, ranges=g.RANGES)
    t0 = time.perf_counter()
    policy, _ = pqn.train(
        (make or ARCHS[arch])(k_init),
        jnp.asarray(c.thetas),
        jnp.asarray(c.y0s),
        k_train,
        cfg or pqn.PQNConfig(),
        replace(ep or g.EP, drop_rate=train_drop),
        RewardConfig(exposure_penalty=w),
    )
    return policy, time.perf_counter() - t0


def save_policy(policy, pol_dir: Path, arch: str, w: float, seed: int) -> None:
    pol_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(pol_dir / f"{arch}_w{w}_s{seed}.eqx", policy)


def train_cell(arch: str, w: float, seed: int, pol_dir: Path | None = None) -> dict:
    policy, secs = train_policy(arch, w, seed)
    out = {"arch": arch, "w": w, "seed": seed, "train_seconds": secs}
    out.update(evaluate(policy, seed, w))
    save_policy(policy, pol_dir or g.OUT / "w4_policies", arch, w, seed)
    return out


def zs_cell(arch: str, w: float, seed: int) -> dict:
    """A W3c policy on the W4 test. Refuses to score it if drop 0 != W3c Tier 0."""
    like = ARCHS[arch](jax.random.PRNGKey(0))
    policy = eqx.tree_deserialise_leaves(g.OUT / "w3c_policies" / f"{arch}_w{w}_s{seed}.eqx", like)
    out = {"arch": arch, "w": w, "seed": seed, **evaluate(policy, seed, w)}
    ref = json.loads((g.OUT / "w3c_shards" / f"{arch}_w{w}_s{seed}.json").read_text())["tier0"]
    for k in ("ttp_capped", "exposure"):
        if out["0.0"][k] != ref[k]:
            raise RuntimeError(f"zs {arch} w{w} s{seed}: drop-0 {k} differs from W3c Tier 0")
    return out


def rules_cell(kind: str, seed: int) -> dict:
    """A rule's trajectory does not depend on w; U is formed per w at analysis."""
    return {"arch": kind, "seed": seed, **evaluate(RulePolicy(kind), seed, 0.0)}


def spawn(kind: str, arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", kind, arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {kind:5s} {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


# ---------------------------------------------------------------- analysis


def utility(cell: dict, p: float, w: float) -> np.ndarray:
    d = cell[str(p)]
    return np.minimum(np.asarray(d["ttp_capped"]), HORIZON) - w * np.asarray(d["exposure"])


def validity(cells: list[dict], archs, seeds) -> tuple[dict, dict]:
    """(valid[(arch, w, seed)], arm_valid[arch]) -- W3c's per-cell rule at drop 0."""
    valid = {(c["arch"], c["w"], c["seed"]): c["closed_loop"] >= g.CLOSED_LOOP_MIN for c in cells}
    arm_valid = {}
    for a in archs:
        per_w = [sum(valid[(a, w, s)] for s in seeds) / len(seeds) for w in WEIGHTS]
        excluded = sum(not valid[(a, w, s)] for w in WEIGHTS for s in seeds)
        arm_valid[a] = (
            min(per_w) >= MIN_VALID_FRACTION
            and excluded / (len(WEIGHTS) * len(seeds)) <= MAX_EXCLUDED
        )
    return valid, arm_valid


def comparison(cells, valid, a, b, stat, seeds, drops) -> dict:
    return delta_ci(paired(cells, valid, a, b, stat, seeds, drops), drops)


def paired(cells, valid, a, b, stat, seeds, drops) -> np.ndarray:
    """(n_w, n_p, n_seed, N_TEST) paired stat_a - stat_b; NaN where a cell is excluded."""
    by = {(c["arch"], c["w"], c["seed"]): c for c in cells}

    def f(c, p, w):
        u = utility(c, p, w)
        return u - utility(c, 0.0, w) if stat == "degradation" else u

    out = np.full((len(WEIGHTS), len(drops), len(seeds), g.N_TEST), np.nan)
    for i, w in enumerate(WEIGHTS):
        for j, p in enumerate(drops):
            for k, s in enumerate(seeds):
                if valid[(a, w, s)] and valid[(b, w, s)]:
                    out[i, j, k] = f(by[(a, w, s)], p, w) - f(by[(b, w, s)], p, w)
    return out


def delta_ci(d: np.ndarray, drops, seed: int = 0) -> dict:
    """Mean over (w, p) of the pooled paired mean, with two bootstrap CIs.

    `ci95` (primary) resamples SEEDS and patients: the training run is the unit
    an architecture claim generalises over. 2026-09-25: one ODE-RNN cell
    retrained under a 2e-7 float perturbation moved its Tier-0 mean U from 132
    to 256 d, and W3c's ODE - LSTM patient-only CI [-20, -11] widens to [-43,
    +23] once seeds are resampled (RESEARCH_LOG, W3c re-analysis). A seed index
    is resampled jointly across w and both arms: cell (w, s) of either arm
    shares seed s's training cohort and eval key. `ci95_patients` is W3c's
    patient-only CI, conditional on the trained policies, kept for continuity.
    """
    est_wp = np.nanmean(d, axis=(2, 3))
    rng = np.random.default_rng(seed)
    pi = rng.integers(0, g.N_TEST, (N_BOOT, g.N_TEST))
    si = rng.integers(0, d.shape[2], (N_BOOT, d.shape[2]))
    both = np.array(
        [np.nanmean(np.nanmean(d[:, :, s][..., i], axis=(2, 3))) for s, i in zip(si, pi)]
    )
    patients = np.nanmean(np.nanmean(d[..., pi], axis=(2, 4)), axis=(0, 1))
    return {
        "delta": float(np.nanmean(est_wp)),
        "ci95": np.percentile(both, [2.5, 97.5]).tolist(),
        "ci95_patients": np.percentile(patients, [2.5, 97.5]).tolist(),
        "per_seed": np.nanmean(d, axis=(0, 1, 3)).tolist(),
        "per_w_p": {
            f"{w}|{p}": float(v) for w, row in zip(WEIGHTS, est_wp) for p, v in zip(drops, row)
        },
    }


def wilcoxon_per_w_p(cells, valid, a, b, seeds, drops=DROPS) -> dict:
    """One-sided (a > b) paired Wilcoxon on U, per (w, p), pooled over valid seeds."""
    by = {(c["arch"], c["w"], c["seed"]): c for c in cells}
    out = {}
    for w in WEIGHTS:
        for p in drops:
            ok = [s for s in seeds if valid[(a, w, s)] and valid[(b, w, s)]]
            if ok:
                ua = np.concatenate([utility(by[(a, w, s)], p, w) for s in ok])
                ub = np.concatenate([utility(by[(b, w, s)], p, w) for s in ok])
                out[f"{w}|{p}"] = paired_wilcoxon(ua, ub, alternative="greater")["p_value"]
    return out


def table(cells, valid, rules, archs, seeds, drops=DROPS) -> dict:
    """Per (policy, w, p), valid cells only: mean U, and the PROJECT.md 10 block --
    capped median TTP + IQR, median exposure, median burden integral -- over
    the pooled patient x seed values."""
    tab = {}
    for w in WEIGHTS:
        for p in drops:
            groups = {a: [c for c in cells if c["arch"] == a and c["w"] == w] for a in archs}
            groups = {a: [c for c in cs if valid[(a, w, c["seed"])]] for a, cs in groups.items()}
            groups.update(
                {r: [c for c in rules if c["arch"] == r and c["seed"] in seeds] for r in RULES}
            )
            for name, cs in groups.items():
                if not cs:
                    continue
                pool = {
                    k: np.concatenate([c[str(p)][k] for c in cs])
                    for k in ("ttp_capped", "censored", "exposure", "burden_integral")
                }
                tab[f"{name}|{w}|{p}"] = {
                    "U": float(np.mean([utility(c, p, w).mean() for c in cs])),
                    "ttp_median": float(np.median(pool["ttp_capped"])),
                    "ttp_iqr": np.percentile(pool["ttp_capped"], [25, 75]).tolist(),
                    "rmst": float(np.mean(pool["ttp_capped"])),
                    "censoring": float(np.mean(pool["censored"])),
                    "exposure_median": float(np.median(pool["exposure"])),
                    "burden_integral_median": float(np.median(pool["burden_integral"])),
                    "n_seeds": len(cs),
                }
    return tab


def analyse(train, zs, rules) -> dict:
    sets = {
        "train": (train, tuple(ARCHS), SEEDS),
        "zs": (zs, ZS_ARCHS, ZS_SEEDS),
    }
    res = {"prereg": "adaptive_therapy/PREREG.md W4 [2026-09-25]"}
    checks = {}
    for name, (cells, archs, seeds) in sets.items():
        valid, arm_valid = validity(cells, archs, seeds)
        checks[name] = (cells, valid, arm_valid, seeds)
        res[f"{name}_excluded"] = [f"{a}_w{w}_s{s}" for (a, w, s), ok in valid.items() if not ok]
        res[f"{name}_arm_valid"] = arm_valid
        res[f"{name}_table"] = table(cells, valid, rules, archs, seeds)
    for h, (name, stat, others) in HYPOTHESES.items():
        cells, valid, arm_valid, seeds = checks[name]
        r = {"valid": all(arm_valid[x] for x in ("ode_rnn", *others)), "vs": {}}
        for x in others:
            r["vs"][x] = comparison(cells, valid, "ode_rnn", x, stat, seeds, TEST_DROPS)
            if stat == "level":  # the drop-0 reference and per-(w, p) tests
                r["vs"][x]["drop0"] = comparison(cells, valid, "ode_rnn", x, stat, seeds, (0.0,))
                r["vs"][x]["wilcoxon_greater_p"] = wilcoxon_per_w_p(
                    cells, valid, "ode_rnn", x, seeds
                )
        r["passes"] = bool(r["valid"] and min(v["ci95"][0] for v in r["vs"].values()) > 0)
        res[h] = r
    cells, valid, _, seeds = checks["train"]  # descriptive: does dt help a discrete arm?
    res["dt_input_effect"] = comparison(cells, valid, "lstm_dt", "lstm", "level", seeds, TEST_DROPS)
    return res


COLORS = {"mlp": "#1F3B4D", "lstm": "#8E7DBE", "lstm_dt": "#E09F3E", "ode_rnn": "#2A9D8F"}
MARKERS = {"mlp": "o", "lstm": "s", "lstm_dt": "^", "ode_rnn": "D"}
RULE_STYLE = {"none": ":", "MTD": "--", "AT50": "-.", "thr0.8": (0, (5, 1, 1, 1))}


W4_ROWS = (
    ("train", "trained with 35% of visits missed"),
    ("zs", "trained on regular visits (W3c)"),
)


def plot(res: dict, path: Path, rows=W4_ROWS, title: str = "W4") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(WEIGHTS), figsize=(13, 6.4), sharex=True, sharey="col")
    for (name, title), ax_row in zip(rows, axes, strict=True):
        tab = res[f"{name}_table"]
        for ax, w in zip(ax_row, WEIGHTS, strict=True):
            for r, ls in RULE_STYLE.items():
                ys = [tab.get(f"{r}|{w}|{p}", {}).get("U", np.nan) for p in DROPS]
                ax.plot(DROPS, ys, ls=ls, lw=1.2, color="#8A8A8A", label=r)
            for a, col in COLORS.items():
                ys = [tab.get(f"{a}|{w}|{p}", {}).get("U", np.nan) for p in DROPS]
                if np.isfinite(ys).any():
                    ax.plot(DROPS, ys, lw=2, color=col, marker=MARKERS[a], ms=6, label=a)
            ax.set_title(f"w = {w}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E6E6E6", lw=0.8)
            ax.set_xticks(DROPS, [f"{p:.0%}" for p in DROPS])
        ax_row[0].set_ylabel(f"{title}\nmean U (days)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("visits missed at test")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=8)
    fig.suptitle(f"{title}: U = min(TTP, 357) − w · exposure, Tier-0 patients", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = (
        [("train", a, w, s) for a in ARCHS for w in WEIGHTS for s in SEEDS]
        + [("zs", a, w, s) for a in ZS_ARCHS for w in WEIGHTS for s in ZS_SEEDS]
        + [("rule", r, 0.0, s) for r in RULES for s in SEEDS + ZS_SEEDS]
    )
    with ThreadPoolExecutor(wf.N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    split = {
        k: [c for j, c in zip(jobs, cells, strict=True) if j[0] == k]
        for k in ("train", "zs", "rule")
    }
    res = analyse(split["train"], split["zs"], split["rule"])
    for h in HYPOTHESES:
        r = res[h]
        print(
            f"{h}: "
            + "  ".join(
                f"ODE−{x} {v['delta']:+.1f} d [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                for x, v in r["vs"].items()
            )
            + f"  valid={r['valid']} PASS={r['passes']}"
        )
    d = res["dt_input_effect"]
    print(f"LSTM+dt − LSTM: {d['delta']:+.1f} d [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
    for name in ("train", "zs"):
        print(f"{name}: arm valid {res[f'{name}_arm_valid']}  excluded {res[f'{name}_excluded']}")
    (g.OUT / "w4_irregular.json").write_text(json.dumps(res, indent=2))
    plot(res, g.OUT / "w4_irregular.png")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        kind, arch, w, seed = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        fn = {
            "train": lambda: train_cell(arch, w, seed),
            "zs": lambda: zs_cell(arch, w, seed),
            "rule": lambda: rules_cell(arch, seed),
        }[kind]
        (SHARDS / f"{kind}_{arch}_w{w}_s{seed}.json").write_text(json.dumps(fn()))
        raise SystemExit(0)
    raise SystemExit(main())
