"""W6: the paper's figures and numbers, read from the logged artifacts only.

Nothing here trains or re-scores a policy; every value is read from a JSON
that a registered run wrote, or from its saved policies (the MLP rule).

  fig_forest.pdf   Delta U with seed+patient 95% CIs, Tiers 0 / 1 / 4, both
                   theta regimes (S1 = Strobl 2021 ranges, L703; provisional =
                   W3c L596 and W4b L668).
  fig_missed.pdf   mean U over w against the missed-visit rate, per arm and
                   the MTD / thr0.8 rules, both regimes.
  fig_rule.pdf     mean greedy dose of the trained MLP policies against
                   burden / baseline, both regimes.
  numbers.tex      \\newcommand macros for every S1 / S2 / W4b / W3c number in the text.

Palette: the dataviz reference palette's first four categorical slots in
their documented order (adjacent-pair gates pass in light mode; slots 3-4 sit
below 3:1 contrast, so every series is direct-labelled). Regime is encoded by
filled vs hollow marks in neutral ink, never by hue, so a hue always means
one arm. 2026-09-29: the palette validator needs node, absent on this box.

Run: JAX_PLATFORMS=cpu uv run python research/w6_figures.py -> paper/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w4_irregular as w4  # noqa: E402  (imports phase2_gate: backend env first)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_therapy.policies.base import DOSE_LEVELS  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

g = w4.g
OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARM = {  # dataviz reference palette, slots 1-4, documented order
    "mlp": ("#2a78d6", "MLP"),
    "lstm": ("#eb6834", "LSTM"),
    "ode_rnn": ("#1baf7a", "ODE-RNN"),
    "lstm_dt": ("#eda100", "LSTM+Δt"),
}
RULE = {"MTD": ("--", "MTD"), "thr0.8": ((0, (1, 1.5)), "thr0.8")}
PROVISIONAL_BLUE = "#86b6ef"  # step 250 of slot 1's ramp: the same arm, other regime

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)


def load() -> dict:
    s1 = json.loads((g.OUT / "s1_strobl.json").read_text())
    return {
        "s1": s1,
        "w4b": json.loads((g.OUT / "w4b_itt.json").read_text()),
        "w3c": json.loads((g.OUT / "w3c_seed_variance.json").read_text())["bootstrap"],
        "s2": json.loads((g.OUT / "s2_headroom.json").read_text()),
        "s3": json.loads((g.OUT / "s3_budget.json").read_text()),
        "m1": json.loads((g.OUT / "m1_monitoring.json").read_text()),
        "m1b": json.loads((g.OUT / "m1b_horizon.json").read_text()),
        "m1c": json.loads((g.OUT / "m1c_medians.json").read_text()),
        "w5pre": json.loads((g.OUT / "w5_precondition.json").read_text()),
    }


def forest_rows(d: dict) -> dict:
    """{tier: {comparison: {"strobl": (delta, lo, hi), "provisional": ...}}}."""

    def t(v):
        return (v["delta"], *v["ci95"])

    A, B = d["s1"]["A"], d["s1"]["B"]
    rows = {}
    for tier in ("tier0", "tier1"):
        rows[tier] = {
            "ODE − MLP": {"strobl": t(A["H_ct_S"]["vs"]["mlp"][tier])},
            "ODE − LSTM": {
                "strobl": t(A["H_ct_S"]["vs"]["lstm"][tier]),
                "provisional": t(d["w3c"][f"ode_rnn-lstm|{tier}"]),
            },
            "LSTM − MLP": {
                "strobl": t(A["memory"][tier]),
                "provisional": t(d["w3c"][f"lstm-mlp|{tier}"]),
            },
        }
    rows["tier4"] = {
        name: {
            "strobl": t(B["H_ct4b"]["vs"][x]),
            "provisional": t(d["w4b"]["H_ct4b"]["vs"][x]),
        }
        for name, x in (
            ("ODE − MLP", "mlp"),
            ("ODE − LSTM", "lstm"),
            ("ODE − LSTM+Δt", "lstm_dt"),
            ("ODE − MTD", "MTD"),
        )
    }
    return rows


def style(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def fig_forest(d: dict, path: Path) -> None:
    rows = forest_rows(d)
    order = ["ODE − MLP", "ODE − LSTM", "ODE − LSTM+Δt", "ODE − MTD", "LSTM − MLP"]
    titles = {
        "tier0": "Tier 0 · in range",
        "tier1": "Tier 1 · parameter shift",
        "tier4": "Tier 4 · 20–50% visits missed",
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), sharex=True, sharey=True)
    for ax, (tier, title) in zip(axes, titles.items(), strict=True):
        for i, name in enumerate(order):
            y = len(order) - 1 - i
            for regime, dy, face in (("strobl", 0.14, INK), ("provisional", -0.14, "white")):
                v = rows[tier].get(name, {}).get(regime)
                if v is None:
                    continue
                est, lo, hi = v
                ax.plot([lo, hi], [y + dy] * 2, color=INK if regime == "strobl" else INK2, lw=1.2)
                ax.plot(est, y + dy, "o", ms=4.5, mfc=face, mec=INK, mew=1.0, zorder=3)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        style(ax)
        ax.set_xlabel("Δ U (days), 95% CI")
    axes[0].set_yticks(range(len(order)), order[::-1])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    handles = [
        plt.Line2D([], [], color=INK, marker="o", mfc=INK, mec=INK, lw=1.2, ms=4.5),
        plt.Line2D([], [], color=INK2, marker="o", mfc="white", mec=INK, lw=1.2, ms=4.5),
    ]
    fig.legend(
        handles,
        ["Strobl 2021 ranges (S1)", "provisional ranges (W3c, W4b)"],
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, metadata={"CreationDate": None})  # byte-reproducible
    plt.close(fig)


def mean_u(table: dict, name: str, key: str) -> float:
    return float(np.mean([table[f"{name}|{w}|{key}"]["U"] for w in w4.WEIGHTS]))


def label_ends(ax, items, x, min_gap) -> None:
    """Direct labels at the right end, nudged apart so none overlap."""
    items = sorted(items, key=lambda it: it[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (_, text), y in zip(items, ys, strict=True):
        ax.text(x, y, text, fontsize=7, color=INK2, va="center")


def fig_missed(d: dict, path: Path) -> None:
    panels = (
        ("Strobl 2021 ranges (S1, seeds 41–50)", d["s1"]["B"]["table"]),
        ("provisional ranges (W4b, seeds 21–30)", d["w4b"]["table"]),
    )
    ps = w4.DROPS
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, (title, tab) in zip(axes, panels, strict=True):
        ends = []
        for r, (ls, label) in RULE.items():
            ys = [mean_u(tab, r, str(p)) for p in ps]
            ax.plot(ps, ys, ls=ls, color=MUTED, lw=1.2)
            ends.append((ys[-1], label))
        for a, (col, label) in ARM.items():
            ys = [mean_u(tab, a, str(p)) for p in ps]
            ax.plot(ps, ys, color=col, lw=1.6, marker="o", ms=4, mec="white", mew=0.8)
            ends.append((ys[-1], label))
        label_ends(ax, ends, ps[-1] + 0.02, 6.5)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xticks(ps, [f"{p:.0%}" for p in ps])
        ax.set_xlim(-0.02, 0.66)
        ax.set_xlabel("visits missed at test")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("mean U over w (days)")
    handles = [plt.Line2D([], [], color=c, lw=1.6, marker="o", ms=4) for c, _ in ARM.values()]
    handles += [plt.Line2D([], [], color=MUTED, lw=1.2, ls=ls) for ls, _ in RULE.values()]
    labels = [lab for _, lab in ARM.values()] + [lab for _, lab in RULE.values()]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, metadata={"CreationDate": None})  # byte-reproducible
    plt.close(fig)


def mlp_mean_dose(pol_dir: Path, seeds, grid: np.ndarray) -> np.ndarray:
    """Mean greedy dose of the saved MLP policies, per grid obs.

    The mean, not the full-dose share: 2026-09-29, past ~1.4x baseline the
    Strobl-regime MLPs step down to HALF dose (78%), which a full-dose share
    draws as "off" (RESEARCH_LOG erratum on U).
    """
    like = w4.ARCHS["mlp"](jax.random.PRNGKey(0))
    levels = np.asarray(DOSE_LEVELS)
    doses = []
    for w in w4.WEIGHTS:
        for s in seeds:
            pol = eqx.tree_deserialise_leaves(pol_dir / f"mlp_w{w}_s{s}.eqx", like)
            q = jax.vmap(lambda o, p=pol: p(o, p.initial_state(), 7.0)[0])(jnp.asarray(grid))
            doses.append(levels[np.asarray(q.argmax(axis=-1))])
    return np.mean(doses, axis=0)


def fig_rule(path: Path) -> dict:
    ratio = np.exp(np.linspace(np.log(0.3), np.log(2.5), 400))
    grid = np.log(ratio)
    dose = {
        "strobl": mlp_mean_dose(g.OUT / "s1_policies" / "a", range(31, 41), grid),
        "provisional": mlp_mean_dose(g.OUT / "w3c_policies", range(6, 11), grid),
    }
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.axvline(0.8, color=MUTED, lw=1.0, ls="--")
    ax.text(0.815, 1.16, "thr0.8 rule", fontsize=7, color=INK2, ha="left")
    ax.plot(ratio, dose["provisional"], color=PROVISIONAL_BLUE, lw=1.6, label="provisional (20)")
    ax.plot(ratio, dose["strobl"], color=ARM["mlp"][0], lw=1.6, label="Strobl 2021 (40)")
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.4)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 0.8, 1.0, 1.5, 2.5], ["0.3", "0.5", "0.8", "1", "1.5", "2.5"])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.28)
    ax.set_yticks([0, 0.5, 1.0], ["0 (off)", "½", "1 (full)"])
    ax.set_xlabel("observed burden / baseline")
    ax.set_ylabel("mean greedy dose,\ntrained MLP policies")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None})  # byte-reproducible
    plt.close(fig)
    on = {k: float(ratio[np.argmax(v >= 0.5)]) for k, v in dose.items()}
    return {"half_full_ratio": on}


MACRO = {  # LaTeX command names cannot hold digits or underscores
    "mlp": "Mlp",
    "lstm": "Lstm",
    "ode_rnn": "Ode",
    "lstm_dt": "LstmDt",
    "MTD": "Mtd",
    "thr0.8": "Thr",
}


def numbers(d: dict, rule: dict) -> str:
    """LaTeX macros: \\name -> "Delta [lo, hi]" (days), from the artifacts."""

    def num(x: float) -> str:
        return "0.0" if abs(x) < 0.05 else f"{x:+.1f}"  # never "+0.0" or "-0.0"

    def fmt(v) -> str:
        est, (lo, hi) = v["delta"], v["ci95"]
        return f"${num(est)}$~[${num(lo)}$, ${num(hi)}$]"

    A, B, P = d["s1"]["A"], d["s1"]["B"], d["w4b"]
    m = {}
    for tier, tag in (("tier0", "Zero"), ("tier1", "One")):
        m[f"SAOdeMlp{tag}"] = fmt(A["H_ct_S"]["vs"]["mlp"][tier])
        m[f"SAOdeLstm{tag}"] = fmt(A["H_ct_S"]["vs"]["lstm"][tier])
        m[f"SAMem{tag}"] = fmt(A["memory"][tier])
    for x in ("mlp", "lstm", "ode_rnn", "lstm_dt"):  # each arm vs MTD at p 0.5
        m[f"SBFb{MACRO[x]}"] = fmt(B["H_fb"]["vs_mtd_p0.5"][x])
        m[f"WBFb{MACRO[x]}"] = fmt(P["H_fb"]["vs_mtd_p0.5"][x])
    for src, tag in ((B, "SB"), (P, "WB")):
        for x, name in (("mlp", "Mlp"), ("lstm", "Lstm"), ("lstm_dt", "LstmDt"), ("MTD", "Mtd")):
            m[f"{tag}Ode{name}"] = fmt(src["H_ct4b"]["vs"][x])
        for x, name in (("mlp", "Mlp"), ("lstm", "Lstm"), ("lstm_dt", "LstmDt")):
            m[f"{tag}Grace{name}"] = fmt(src["H_grace"]["vs"][x])
        fb = src["H_fb"]
        m[f"{tag}FbBest"] = {"mlp": "MLP", "lstm": "LSTM", "lstm_dt": "LSTM+$\\Delta t$"}.get(
            fb["best"], "ODE-RNN"
        )
        m[f"{tag}Fb"] = fmt(fb["vs_mtd_p0.5"][fb["best"]])
        for a in (*ARM, "MTD", "thr0.8"):
            for p, pn in (("0.0", "Zero"), ("0.2", "Fifth"), ("0.5", "Half")):
                m[f"{tag}U{MACRO[a]}{pn}"] = f"{mean_u(src['table'], a, p):.1f}"
    for tier, tag in (("tier0", "Zero"), ("tier1", "One")):
        for a in ("mlp", "thr0.8", "MTD", "lstm", "ode_rnn"):
            m[f"SAU{MACRO[a]}{tag}"] = f"{mean_u(A['table'], a, tier):.1f}"
    for tier, tag in (("tier0", "Zero"), ("tier1", "One")):  # W3c, seed+patient CIs (L596)
        m[f"WCOdeLstm{tag}"] = fmt(d["w3c"][f"ode_rnn-lstm|{tier}"])
        m[f"WCMem{tag}"] = fmt(d["w3c"][f"lstm-mlp|{tier}"])
    for k, v in rule["half_full_ratio"].items():
        m[f"RuleHalf{k.capitalize()}"] = f"{v:.2f}"
    S2 = d["s2"]  # the full-information reference (PREREG S2), never an arm
    for key, name in (
        ("o_full-mlp|tier0", "HFullZero"),
        ("o_full-mlp|tier1", "HFullOne"),
        ("o_state-mlp|tier0", "HStateZero"),
        ("o_state-mlp|tier1", "HStateOne"),
    ):
        m[name] = fmt(S2["A"][key])
    for key, name in (
        ("o_full-MTD|0.5", "HFullMtdHalf"),
        ("o_full-lstm|0.5", "HFullLstmHalf"),
        ("o_full-mlp|test", "HFullMlpTest"),
        ("o_state-MTD|0.5", "HStateMtdHalf"),
    ):
        m[name] = fmt(S2["B"][key])
    S3 = d["s3"]  # the headroom gate under PQN's published parallelism
    m["SThreeFull"] = fmt(S3["primary"])
    m["SThreeFullOne"] = fmt(S3["tier1"])
    m["SThreeBudgetMlp"] = fmt(S3["budget_effect"]["mlp"])
    m["SThreeBudgetFull"] = fmt(S3["budget_effect"]["o_full"])
    m["SThreeR"] = f"${S3['R']['mean']:.1f}$"
    m["SThreeRPos"] = f"{S3['R']['cells_positive']}/{S3['R']['n']}"
    for a, name in (("mlp", "Mlp"), ("thr0.8", "Thr")):
        m[f"SThreeU{name}Zero"] = f"{mean_u(S3['table'], a, 'tier0'):.1f}"
    M1 = d["m1"]  # the monitoring map (PREREG M1), rules only
    cost = list(M1["H_cost"].values())
    m["MOneCostLow"], m["MOneCostHigh"] = (fmt(_as_ci(v)) for v in cost)
    m["MOneMiss"] = fmt(M1["H_miss"])
    strobl = [b for b in M1["bins"].values() if not b["outside_strobl_sweep"]]
    for half, name in ((strobl[:4], "Low"), (strobl[4:], "High")):
        ps = [b["hold"]["p_star"] for b in half]
        m[f"MOnePStar{name}Min"], m[f"MOnePStar{name}Max"] = f"{min(ps):.2f}", f"{max(ps):.2f}"
    p0 = [b["hold"]["delta"][0] for b in M1["bins"].values()]
    m["MOneRegMin"], m["MOneRegMax"] = f"{min(p0):+.0f}", f"{max(p0):+.0f}"
    M1b = d["m1b"]  # the same map over three years (PREREG M1b)
    cost = list(M1b["H_cost"].values())
    m["MOneBCostLow"], m["MOneBCostHigh"] = (fmt(_as_ci(v)) for v in cost)
    m["MOneBMiss"] = fmt(M1b["H_miss"])
    sb = [b for b in M1b["bins"].values() if not b["outside_strobl_sweep"]]
    low = [b["hold"]["p_star"] for b in sb[:4]]
    m["MOneBPStarLowMin"], m["MOneBPStarLowMax"] = f"{min(low):.2f}", f"{max(low):.2f}"
    high = [b["hold"]["p_star"] for b in sb[4:]]
    m["MOneBPStarHigh"] = "above 0.6" if all(p == ">0.6" for p in high) else str(high)
    half = [b["hold"]["delta"][5] for b in sb[4:]]  # p 0.5
    m["MOneBHighMin"], m["MOneBHighMax"] = f"{min(half):+.0f}", f"{max(half):+.0f}"
    tail = sb[-1]["p0.5_hold_standing"]  # no-cost end, high turnover
    m["MOneBTailCens"] = f"{tail['best_thr']['censoring']:.0%}".replace("%", "\\%")
    m["MOneBMedThr"] = f"{tail['best_thr']['ttp_median']:.0f}"
    m["MOneBMedMtd"] = f"{tail['MTD']['ttp_median']:.0f}"
    cut = 1 - tail["best_thr"]["exposure_median"] / tail["MTD"]["exposure_median"]
    m["MOneBExpCut"] = f"{cut:.0%}".replace("%", "\\%")
    M1c = d["m1c"]  # both maps under medians (PREREG M1c, post-hoc)
    hb = [b for b in M1c["M1b"]["bins"].values() if not b["outside_strobl_sweep"]][4:]
    for b, name in zip(hb, ("One", "Two", "Three", "Four"), strict=True):  # costliest first
        m[f"MOneCHigh{name}"] = fmt({"delta": b["hold"]["delta_med"][5], "ci95": b["p0.5_ci95"]})
    m["MOneCCostOneLow"] = fmt(_as_ci(list(M1c["M1"]["H_cost"].values())[0]))
    st = [b["p0.5_standing"] for b in hb]
    up = [s["best_thr"]["burden_integral_median"] / s["MTD"]["burden_integral_median"] for s in st]
    m["MOneCBurdenNoCost"] = f"{up[-1] - 1:.0%}".replace("%", "\\%")
    m["MOneCBurdenCostly"] = f"{up[0]:.1f}"
    W5 = d["w5pre"]["families"]  # the other two families' precondition (PREREG W5-pre)
    hf, kz = W5["hahnfeldt"]["357d_p0.0"], W5["kuznetsov"]["357d_p0.0"]
    m["WFiveHfTtp"] = f"{hf['MTD']['ttp_median']:.0f}"
    kthr = [v["ttp_median"] for k, v in kz.items() if k.startswith("thr")]
    m["WFiveKzThrMin"], m["WFiveKzThrMax"] = f"{min(kthr):.0f}", f"{max(kthr):.0f}"
    m["WFiveKzMtdCens"] = f"{kz['MTD']['censoring']:.0%}".replace("%", "\\%")
    head = "% GENERATED by research/w6_figures.py from research/outputs/*.json -- do not edit\n"
    return head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items()))


def _as_ci(v: dict) -> dict:
    """M1's H_cost entries -> the {delta, ci95} shape `fmt` expects."""
    return {"delta": v["high_cost_minus_no_cost"], "ci95": v["ci95"]}


COST_RAMP = ("#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")  # ordinal, high cost darkest


def fig_monitoring(d: dict, path: Path) -> None:
    """Delta_fb (hold) against missed-visit rate, per cost bin; turnover x horizon panels."""
    ps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True)
    for row, (key, horizon) in enumerate((("m1", "1-year horizon"), ("m1b", "3-year horizon"))):
        bins = d[key]["bins"]
        names = list(bins)
        for col, (half, title) in enumerate(
            (
                (names[:5], "low turnover (d_T / r_S < 0.25)"),
                (names[5:], "high turnover (d_T / r_S ≥ 0.25)"),
            )
        ):
            ax = axes[row, col]
            ax.axhline(0, color=INK, lw=0.8)
            for name, colour in zip(half, COST_RAMP, strict=True):
                b = bins[name]
                ax.plot(
                    ps,
                    b["hold"]["delta"],
                    color=colour,
                    lw=1.6,
                    ls="--" if b["outside_strobl_sweep"] else "-",
                    marker="o",
                    ms=3.5,
                    mec="white",
                    mew=0.6,
                )
            if row == 0:
                ax.set_title(title, fontsize=8, color=INK, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
        axes[row, 0].set_ylabel(f"{horizon}\nΔU vs MTD (days)", fontsize=7.5)
        axes[row, 1].sharey(axes[row, 0])
    for ax in axes[1]:
        ax.set_xticks(ps, [f"{p:.0%}" for p in ps])
        ax.set_xlabel("visits missed (a missed visit repeats the last dose)")
    labels = [
        "0.25–0.5 (outside Strobl's sweep)",
        "0.5–0.625",
        "0.625–0.75",
        "0.75–0.875",
        "0.875–1",
    ]
    handles = [
        plt.Line2D([], [], color=c, lw=1.6, ls="--" if i == 0 else "-")
        for i, c in enumerate(COST_RAMP)
    ]
    fig.legend(
        handles,
        labels,
        title="r_R / r_S (darker = costlier resistance)",
        title_fontsize=7.5,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, metadata={"CreationDate": None})  # byte-reproducible
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    fig_forest(d, OUT / "fig_forest.pdf")
    fig_missed(d, OUT / "fig_missed.pdf")
    rule = fig_rule(OUT / "fig_rule.pdf")
    fig_monitoring(d, OUT / "fig_monitoring.pdf")
    (OUT.parent / "numbers.tex").write_text(numbers(d, rule))
    print(
        f"wrote {OUT}/fig_{{forest,missed,rule,monitoring}}.pdf and paper/numbers.tex; rule {rule}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
