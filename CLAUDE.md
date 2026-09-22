# neural-ode-adaptive-therapy

Continuous-time RL policies for robust cancer adaptive therapy. Does structuring
a policy as a differential equation give it an inductive bias for tumour dynamics
that survives when the biology is unknown, drifting, or from an unseen mechanism?

    virtual patient (model_id, θ) → diffrax rollout → burden-only observation
                                  → policy (MLP | LSTM | ODE-RNN | LTC) → dose

## Session start — always do this

1. **Read [LAWS.md](LAWS.md) first.** It is the cheapest gate in the repo; most
   proposals die there in ten lines. Then [CLOSED.md](CLOSED.md) if the idea
   survives.
2. **Read [adaptive_therapy/VERDICTS.md](adaptive_therapy/VERDICTS.md)** — 30
   lines, not the log.
3. Open [RESEARCH_LOG.md](adaptive_therapy/RESEARCH_LOG.md) **only** at the
   entries a verdict links to. Do not read whole logs by default.
4. If the session will touch any doc, run the audit first:

       make audit          # or: uv run python tools/doc_audit.py -v

   **FAIL (exit 1)** means a link is broken and must be fixed before committing.
   **WARN** is backlog. These rules were honour-system in the sibling repo for
   months and drifted badly — do not go back to trusting the rule instead of the
   check.

## Precedence ladder — when two docs disagree

Docs contradict each other as a program moves. There is one ladder, and it is
global (not re-negotiated per file). **Higher wins, always:**

1. **[adaptive_therapy/RESEARCH_LOG.md](adaptive_therapy/RESEARCH_LOG.md)** —
   the append-only archive. Nothing overrides a logged result.
2. **[LAWS.md](LAWS.md) / [CLOSED.md](CLOSED.md)** — generalizations bought with
   ≥2 confirmations, and the kill list. On conflict with the log, fix these.
3. **[PROJECT.md](PROJECT.md)** — the spec. Read it; **do not edit it to match a
   result.** A result that contradicts the spec is a log entry, then a verdict.
4. **[ROADMAP.md](ROADMAP.md), [PREREG.md](adaptive_therapy/PREREG.md)** —
   committed plans.
5. **This file and the skills** — lowest rung; they go stale fastest.

A doc at a lower rung asserting something about a higher rung is a **staleness
bug in the lower doc** — fix it there, and never "resolve" it by editing upward.
A law that contradicts a logged result is a bug in the law.

[literature/](literature/) sits **below everything**: a paper is not evidence
about our simulators. It says what to test and what to control for, never what
is true here.

## The research record

| File | Rung | Rule |
|---|---|---|
| `adaptive_therapy/RESEARCH_LOG.md` | 1 | Append-only forever. ≤40 lines/entry, hard cap. Newest at bottom. |
| `adaptive_therapy/VERDICTS.md` | 2 | The index. 3 sections; one line + `[Lnnnn](RESEARCH_LOG.md#Lnnnn)` per verdict. |
| `CLOSED.md` / `LAWS.md` | 2 | Program-level closures; a law needs ≥2 independent confirmations. |
| `ROADMAP.md` | 4 | Phases with kill criteria. Struck through + linked when killed, never deleted. |
| `adaptive_therapy/PREREG.md` | 4 | Hypothesis + exact pass/kill criteria, written *before* the run. Append-only. |

**Entry format** (the `**Verdict:**` line is mandatory and machine-greppable
via `grep -A1 "^## \[" adaptive_therapy/RESEARCH_LOG.md`):

    ## [YYYY-MM-DD] — One-line summary
    **Verdict:** OPEN|FROZEN|CLOSED — half-line takeaway · **Gate:** median TTP + IQR / exposure / 2nd endpoint / Wilcoxon p / n seeds · **Artifacts:** paths
    ### Context / What was tried / Results / Decision

The **Gate** field carries [PROJECT.md §10](PROJECT.md)'s metrics, so a
single-seed or mean-not-median result is visibly non-conforming *in the log
itself*. **≤40 lines is a hard cap, not a target** — if an entry will not fit,
the run needed an artifact, not more prose.

**After appending an entry, update VERDICTS.md in the same commit.** The log is
the archive; VERDICTS is the index. An entry without an index update is
invisible to future sessions.

## Promotion chain

    run → RESEARCH_LOG entry → VERDICTS line → (cross-cutting) CLOSED item
        → (≥2 closures, same mechanism) LAW

## Module map

**This table says what each path *is*, never what its research currently
concludes.** A verdict copied here is a second copy nobody updates and every
session reads as truth.

| Path | Role |
|---|---|
| `adaptive_therapy/envs/` | The three ODE families (LV, Hahnfeldt, Kuznetsov) + PK + reward. Pure functions, no I/O. |
| `adaptive_therapy/policies/` | Actor architectures: `mlp`, `lstm`, `ode_rnn`, `ltc`. The ablation axis. |
| `adaptive_therapy/rl/` | One RL algorithm (PPO/A3C). The architecture is swapped *inside* it; everything else held fixed. |
| `adaptive_therapy/eval/` | Robustness tiers, TTP (incl. the non-monotone rule), exposure, paired Wilcoxon. |
| `adaptive_therapy/tests/` | pytest. Test names state the invariant, not the function. |
| `research/` | One-shot scripts and **all** I/O. Never imported by the package. Writes to gitignored `research/outputs/`. |
| `literature/` | Paper reviews with enforced provenance. See `REVIEW_PROTOCOL.md`. |
| `tools/doc_audit.py` | Record-integrity check. Hard-fails on broken links. |

## Non-negotiables (from the spec)

- **Burden-only observation.** Never subpopulations — that is what makes it a POMDP.
- **Report exposure with TTP, always.** Winning TTP by dosing constantly is not robustness.
- **Medians + IQR, never means.** Trajectories are heavy-tailed.
- **Paired Wilcoxon** across matched virtual patients, never a t-test.
- **Multiple seeds with variance bands.** Single-seed is a desk-reject.
- **Strict 2×2** for architecture × curriculum. No confounded comparisons.
- **No citation without a resolving DOI/venue**, and a `2°` (unread) source may
  not be cited in a design argument.
- **No clinical-outcome claims.** Virtual patients only in v1.

## Conventions

- Python via **uv**: `make setup`, then everything runs as `uv run …`.
- `make all` = lint + test + audit + lit-check. Pre-commit runs the same checks.
- Commits: `[research]`, `[feat]`, `[fix]`, `[chore]`, `[docs]`. Subject states
  the *finding*, not the file list. Stage only the result's own files; never
  `git add -A`. Do not push without being asked.
- Dated, evidenced comments on every config exception — *why*, with the incident.
- Seeds threaded explicitly as parameters (`seed: int = SEED`), never global.
