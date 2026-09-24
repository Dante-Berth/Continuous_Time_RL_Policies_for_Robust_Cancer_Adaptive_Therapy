"""Figures for the progress deck: dosing regimes on the three tumour models.

Illustration only -- not pre-registered, produces no verdict. Every regime runs
through the SAME closed-loop rollout (rl/rollout.py) the trained policies use,
so a rule and a network differ only in the policy object:

  none       never treat
  MTD        maximum dose at every visit (standard of care)
  AT50       Zhang et al.'s trial rule: treat until burden < 50% of baseline,
             hold until it is back at baseline (hysteresis, carried as state)
  thr0.8     treat above 0.8x baseline -- the rule Gallagher 2024's A3C learned
             (N* = 0.83 n0) and our PQN MLP learned in W1 (L461)
  MLP / LSTM / ODE-RNN   the trained LV policies (w 0.05, seed 0). Their input
             is a scale-free log-ratio, so they can be RUN on Hahnfeldt and
             Kuznetsov too; they were never trained there.

Run: JAX_PLATFORMS=cpu uv run python research/deck_figures.py
     -> research/outputs/deck/*.png, deck/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_gate as g  # noqa: E402  (sets the backend env before jax work)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adaptive_therapy.envs.reward import RewardConfig  # noqa: E402
from adaptive_therapy.envs.sampling import sample_cohort  # noqa: E402
from adaptive_therapy.eval.metrics import cumulative_exposure, ttp_running_nadir  # noqa: E402
from adaptive_therapy.policies.lstm import LSTMPolicy  # noqa: E402
from adaptive_therapy.policies.mlp import MLPPolicy  # noqa: E402
from adaptive_therapy.policies.ode_rnn import ODERNNPolicy  # noqa: E402
from adaptive_therapy.policies.rules import RulePolicy  # noqa: E402
from adaptive_therapy.rl.rollout import EpisodeConfig, batch_rollout  # noqa: E402

OUT = g.OUT / "deck"
FAMILIES = {
    "lotka_volterra": "Lotka–Volterra (competition)",
    "hahnfeldt": "Hahnfeldt (angiogenesis)",
    "kuznetsov": "Kuznetsov (immune predation)",
}
N_COHORT = 100
COLORS = {
    "none": "#9A9A9A",
    "MTD": "#B23A48",
    "AT50": "#E09F3E",
    "thr0.8": "#3C6E71",
    "MLP": "#284B63",
    "LSTM": "#8E7DBE",
    "ODE-RNN": "#2A9D8F",
}


def trained():
    k = jax.random.PRNGKey(0)
    load = lambda d, name, like: eqx.tree_deserialise_leaves(g.OUT / d / name, like)  # noqa: E731
    return {
        "MLP": load("phase2c_policies", "mlp_w0.05_s0.eqx", MLPPolicy(k, 441, True)),
        "LSTM": load("w3_policies", "lstm_w0.05_s0.eqx", LSTMPolicy(k, layer_norm=True)),
        "ODE-RNN": load(
            "phase2c_policies", "ode_rnn_w0.05_s0.eqx", ODERNNPolicy(k, layer_norm=True)
        ),
    }


def run(fam: str, policies: dict):
    ep = EpisodeConfig(n_visits=52, visit_interval=7.0, model_id=fam)
    ts = np.arange(ep.n_visits) * ep.visit_interval
    c = sample_cohort(fam, N_COHORT, seed=0)
    th, y0 = jnp.asarray(c.thetas), jnp.asarray(c.y0s)
    out = {}
    for name, pol in policies.items():
        tr = batch_rollout(pol, th, y0, jax.random.PRNGKey(0), ep, RewardConfig(), greedy=True)
        burden, dose = np.asarray(tr["burden"]), np.asarray(tr["dose"])
        ttp = ttp_running_nadir(burden, ts)
        out[name] = {
            "burden": burden / burden[:, :1],
            "dose": dose,
            "ttp": ttp.days,
            "exposure": cumulative_exposure(dose, ts),
        }
    return ts, out


def pick_patient(res) -> int:
    """The patient whose untreated TTP is closest to the cohort median."""
    t = res["none"]["ttp"]
    return int(np.argmin(np.abs(t - np.median(t))))


def fig_regimes(fam, ts, res, path):
    i = pick_patient(res)
    names = list(res)
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.4], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    for n in names:
        ax.plot(
            ts,
            res[n]["burden"][i],
            color=COLORS[n],
            lw=2.2 if n != "none" else 1.6,
            ls="--" if n == "none" else "-",
            label=n,
        )
    ax.axhline(1.2, color="#555", lw=0.8, ls=":")
    ax.text(ts[-1], 1.22, "progression (+20%)", ha="right", va="bottom", fontsize=9, color="#555")
    ax.set_yscale("log")
    ax.set_ylabel("burden / baseline (log)")
    ax.tick_params(labelbottom=False)
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 1e-6))  # Kuznetsov: MTD eradicates to ~1e-40
    ax.set_xlim(0, ts[-1])
    ax.legend(
        ncol=len(names), fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    strip = np.stack([res[n]["dose"][i] for n in names])
    ax2.imshow(
        strip,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        extent=(0, ts[-1] + 7, len(names) - 0.5, -0.5),
        interpolation="nearest",
    )
    ax2.set_yticks(range(len(names)), names, fontsize=9)
    ax2.set_xlabel("day")
    ax2.set_title(
        "dose per weekly visit (black = full, grey = half, white = none)", fontsize=9, loc="left"
    )
    fig.suptitle(FAMILIES[fam] + " — one typical virtual patient", x=0.06, ha="left", fontsize=13)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_frontier_summary(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (fam, rows) in zip(axes, summary.items(), strict=True):
        for n, r in rows.items():
            ax.scatter(r["exposure_median"], r["ttp_rmst"], s=70, color=COLORS[n], zorder=3)
            ax.annotate(
                n,
                (r["exposure_median"], r["ttp_rmst"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_title(FAMILIES[fam], fontsize=11)
        ax.set_xlabel("median cumulative exposure")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("restricted-mean TTP (days, 357 cap)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_mlp_rule(path):
    d = json.loads((g.OUT / "w1_diagnose.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, segs in d["mlp_rules"].items():
        xs, ys = [], []
        for s in segs:
            xs += [np.exp(s["obs_from"]), np.exp(s["obs_to"])]
            ys += [s["dose"], s["dose"]]
        ax.step(xs, ys, where="post", lw=1.8, alpha=0.75, label=k.replace("_", " "))
    ax.axvline(0.83, color="#B23A48", ls="--", lw=1.2)
    ax.text(0.85, 0.35, "Gallagher 2024\nN* = 0.83", color="#B23A48", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 2.5)
    ax.set_xticks([0.1, 0.2, 0.5, 0.8, 1, 2], ["0.1", "0.2", "0.5", "0.8", "1", "2"])
    ax.minorticks_off()
    ax.set_xlabel("observed burden / baseline")
    ax.set_ylabel("greedy dose")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    policies = {k: RulePolicy(k) for k in ("none", "MTD", "AT50", "thr0.8")} | trained()
    summary = {}
    for fam in FAMILIES:
        ts, res = run(fam, policies)
        fig_regimes(fam, ts, res, OUT / f"regimes_{fam}.png")
        summary[fam] = {
            n: {
                "ttp_rmst": float(np.mean(r["ttp"])),
                "ttp_median_capped": float(np.median(r["ttp"])),
                "exposure_median": float(np.median(r["exposure"])),
            }
            for n, r in res.items()
        }
        print(
            fam,
            {
                n: (round(v["ttp_rmst"]), round(v["exposure_median"]))
                for n, v in summary[fam].items()
            },
        )
    fig_frontier_summary(summary, OUT / "regimes_summary.png")
    fig_mlp_rule(OUT / "mlp_rule.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
