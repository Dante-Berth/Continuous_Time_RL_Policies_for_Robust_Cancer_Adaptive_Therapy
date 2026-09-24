"""W3: memory vs continuous time on Tiers 0 and 1, on TTP-vs-exposure frontiers.

PRE-REGISTERED in adaptive_therapy/PREREG.md [2026-09-24] W3, before the LSTM
arm existed or any cell of this grid ran. Hypotheses:
  H_mem  Gap(LSTM, MLP) on Tier 1 > 0, 95% CI excluding 0
  H_ct   Gap(ODE-RNN, LSTM) on Tier 1 > 0, 95% CI excluding 0
Validity: every cell greedy observation_dependence >= 0.10; frontiers must
overlap on >= 20% of both arms' exposure ranges. "Robust" wording only if the
Tier-1 gap CI lies above the Tier-0 gap point estimate.

Frontier: per arm/tier/w, the median over seeds of (median cumulative exposure,
capped median TTP); piecewise-linear through the 5 w points sorted by exposure.
Gap = mean y_A - y_B over the exposure overlap (100-point grid). CI = 2000
patient bootstrap resamples, the same indices for every cell.

Grid: {MLP, LSTM, ODE-RNN} x w {0, .05, .10, .20, .40} x seeds {0,1,2} = 45
cells, PQN (Phase-2c trainer, unchanged), each cell in a fresh process.

W3b [2026-09-24] re-registers W3 after it was INVALID (RESEARCH_LOG L496):
fresh seeds {3,4,5}, w = 0 dropped, and the frontier's y-axis is restricted-
mean TTP (mean of min(TTP, horizon)) because W3's capped median sat at the
horizon for every MLP cell. Everything else is W3's rule, unchanged.

Run: JAX_PLATFORMS=cpu uv run python research/w3_frontier.py [w3 | w3b]
     -> research/outputs/<reg>_frontier.{json,png}, <reg>_shards/
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_gate as g  # noqa: E402  (sets the backend env before jax work)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from w1_diagnose import obs_explained, tier0_cohort  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import cumulative_exposure, ttp_running_nadir  # noqa: E402
from adaptive_therapy.policies.lstm import LSTMPolicy  # noqa: E402
from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy  # noqa: E402
from adaptive_therapy.rl import ppo, pqn  # noqa: E402
from adaptive_therapy.rl.rollout import observation_dependence  # noqa: E402

ARCHS = {
    "mlp": lambda k: MLPPolicy(k, width=441, layer_norm=True),
    "lstm": lambda k: LSTMPolicy(k, layer_norm=True),
    "ode_rnn": lambda k: ODERNNPolicy(k, layer_norm=True),
}
# One entry per registration, so each stays reproducible from HEAD.
REGISTRATIONS = {
    "w3": {"weights": (0.0, 0.05, 0.10, 0.20, 0.40), "seeds": (0, 1, 2), "y": np.median},
    # RMST: a mean is a declared deviation from PROJECT.md 10 (PREREG W3b).
    "w3b": {"weights": (0.05, 0.10, 0.20, 0.40), "seeds": (3, 4, 5), "y": np.mean},
}
REG = next((a for a in sys.argv[1:] if a in REGISTRATIONS), "w3")
WEIGHTS = REGISTRATIONS[REG]["weights"]
SEEDS = REGISTRATIONS[REG]["seeds"]
Y_STAT = REGISTRATIONS[REG]["y"]
TIERS = {"tier0": tier0_cohort, "tier1": g.tier1_cohort}
N_BOOT = 2000
MIN_OVERLAP = 0.20
N_PARALLEL = 6  # 4 threads each on a 32-core box
SHARDS = g.OUT / f"{REG}_shards"


def run_cell(arch: str, w: float, seed: int) -> dict:
    k_init, k_train, k_eval = jax.random.split(jax.random.PRNGKey(seed), 3)
    train_c = sample_cohort(g.FAM, g.N_TRAIN, seed=seed)
    rw = RewardConfig(exposure_penalty=w)
    t0 = time.perf_counter()
    policy, _ = pqn.train(
        ARCHS[arch](k_init),
        jnp.asarray(train_c.thetas),
        jnp.asarray(train_c.y0s),
        k_train,
        pqn.PQNConfig(),
        g.EP,
        rw,
    )
    out = {"arch": arch, "w": w, "seed": seed, "train_seconds": time.perf_counter() - t0}
    for tier, cohort in TIERS.items():
        th, y0 = cohort(g.N_TEST, seed)
        ev = ppo.evaluate(policy, jnp.asarray(th), jnp.asarray(y0), k_eval, g.EP, rw)
        burden, dose, obs = (np.asarray(ev[k]) for k in ("burden", "dose", "obs"))
        ttp = ttp_running_nadir(burden, g.TS)
        out[tier] = {
            "ttp_capped": ttp.days.tolist(),  # censored patients sit at the horizon
            "censored": ttp.censored.tolist(),
            "exposure": cumulative_exposure(dose, g.TS).tolist(),
            "closed_loop": observation_dependence(policy, ev["obs"], g.EP, k_eval, greedy=True),
            "dose_explained_by_obs": obs_explained(obs, dose),
        }
    pol_dir = g.OUT / f"{REG}_policies"
    pol_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(pol_dir / f"{arch}_w{w}_s{seed}.eqx", policy)
    return out


def spawn(arch: str, w: float, seed: int) -> dict:
    shard = SHARDS / f"{arch}_w{w}_s{seed}.json"
    if not shard.exists():  # resume: a finished cell is never recomputed
        subprocess.run(
            [sys.executable, __file__, "--worker", REG, arch, str(w), str(seed)],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    print(f"  done {arch:8s} w={w:<4} s{seed}", flush=True)
    return json.loads(shard.read_text())


def frontier(cells: list[dict], tier: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) points of one arm, sorted by exposure, on patient indices `idx`."""
    pts = []
    for w in WEIGHTS:
        per_seed = [c[tier] for c in cells if c["w"] == w]
        x = np.median([np.median(np.asarray(s["exposure"])[idx]) for s in per_seed])
        y = np.median([Y_STAT(np.asarray(s["ttp_capped"])[idx]) for s in per_seed])
        pts.append((x, y))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def gap(fa, fb) -> tuple[float, float]:
    """Mean y_A - y_B over the exposure overlap, and the overlap fraction."""
    lo, hi = max(fa[0].min(), fb[0].min()), min(fa[0].max(), fb[0].max())
    if hi <= lo:
        return float("nan"), 0.0
    xs = np.linspace(lo, hi, 100)
    frac = min(
        (hi - lo) / max(np.ptp(fa[0]), 1e-10),
        (hi - lo) / max(np.ptp(fb[0]), 1e-10),
    )
    return float(np.mean(np.interp(xs, *fa) - np.interp(xs, *fb))), float(frac)


def compare(by_arm: dict, a: str, b: str, tier: str, seed: int = 0) -> dict:
    full = np.arange(g.N_TEST)
    est, frac = gap(frontier(by_arm[a], tier, full), frontier(by_arm[b], tier, full))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, g.N_TEST, g.N_TEST)
        boots.append(gap(frontier(by_arm[a], tier, idx), frontier(by_arm[b], tier, idx))[0])
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan)
    return {"gap_days": est, "ci95": [float(lo), float(hi)], "overlap_fraction": frac}


def plot(by_arm: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    full = np.arange(g.N_TEST)
    for ax, tier in zip(axes, TIERS, strict=True):
        for arch, cells in by_arm.items():
            x, y = frontier(cells, tier, full)
            (line,) = ax.plot(x, y, marker="o", label=arch)
            for c in cells:  # per-seed points, so a median never hides a seed
                ax.scatter(
                    np.median(c[tier]["exposure"]),
                    Y_STAT(c[tier]["ttp_capped"]),
                    s=10,
                    alpha=0.35,
                    color=line.get_color(),
                )
        ax.set_title(tier.replace("tier", "Tier "))
        ax.set_xlabel("median cumulative exposure")
    axes[0].set_ylabel(f"capped {Y_STAT.__name__} TTP (days)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, s) for a in ARCHS for w in WEIGHTS for s in SEEDS]
    with ThreadPoolExecutor(N_PARALLEL) as ex:
        cells = list(ex.map(lambda j: spawn(*j), jobs))
    by_arm = {a: [c for c in cells if c["arch"] == a] for a in ARCHS}

    closed = {f"{c['arch']}_w{c['w']}_s{c['seed']}": c["tier1"]["closed_loop"] for c in cells}
    valid_arm = {
        a: all(c["tier1"]["closed_loop"] >= g.CLOSED_LOOP_MIN for c in cs)
        for a, cs in by_arm.items()
    }
    result = {
        "prereg": f"adaptive_therapy/PREREG.md {REG.upper()} [2026-09-24]",
        "valid_arm": valid_arm,
    }
    for name, (a, b) in {"H_mem": ("lstm", "mlp"), "H_ct": ("ode_rnn", "lstm")}.items():
        r = {t: compare(by_arm, a, b, t) for t in TIERS}
        t1, t0 = r["tier1"], r["tier0"]
        overlap_ok = all(r[t]["overlap_fraction"] >= MIN_OVERLAP for t in TIERS)
        valid = valid_arm[a] and valid_arm[b] and overlap_ok
        r["valid"] = bool(valid)
        r["overlap_ok"] = bool(overlap_ok)
        r["passes"] = bool(valid and t1["ci95"][0] > 0)
        r["robust_wording_allowed"] = bool(r["passes"] and t1["ci95"][0] > t0["gap_days"])
        result[name] = r
        print(
            f"{name} ({a} vs {b}): "
            + "  ".join(
                f"{t}: gap {r[t]['gap_days']:+.1f} d [{r[t]['ci95'][0]:+.1f}, "
                f"{r[t]['ci95'][1]:+.1f}] overlap {r[t]['overlap_fraction']:.0%}"
                for t in TIERS
            )
            + f"  valid={r['valid']} PASS={r['passes']} robust={r['robust_wording_allowed']}"
        )

    full = np.arange(g.N_TEST)
    result["frontiers"] = {
        a: {t: [list(map(float, v)) for v in frontier(cs, t, full)] for t in TIERS}
        for a, cs in by_arm.items()
    }
    result["cells"] = [
        {
            "arch": c["arch"],
            "w": c["w"],
            "seed": c["seed"],
            **{
                t: {
                    "ttp_median_capped": float(np.median(c[t]["ttp_capped"])),
                    "ttp_iqr_capped": np.percentile(c[t]["ttp_capped"], [25, 75]).tolist(),
                    "ttp_rmst": float(np.mean(c[t]["ttp_capped"])),
                    "censoring": float(np.mean(c[t]["censored"])),
                    "exposure_median": float(np.median(c[t]["exposure"])),
                    "closed_loop": c[t]["closed_loop"],
                    "dose_explained_by_obs": c[t]["dose_explained_by_obs"],
                }
                for t in TIERS
            },
        }
        for c in cells
    ]
    result["closed_loop_tier1"] = closed
    print("valid arms:", valid_arm)
    for a, f in result["frontiers"].items():
        for t, (x, y) in f.items():
            print(f"  {a:8s} {t}: " + " ".join(f"({xi:.0f},{yi:.0f})" for xi, yi in zip(x, y)))
    (g.OUT / f"{REG}_frontier.json").write_text(json.dumps(result, indent=2))
    plot(by_arm, g.OUT / f"{REG}_frontier.png")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        arch, w, seed = sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        (SHARDS / f"{arch}_w{w}_s{seed}.json").write_text(json.dumps(run_cell(arch, w, seed)))
        raise SystemExit(0)
    raise SystemExit(main())
