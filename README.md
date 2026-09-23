# neural-ode-adaptive-therapy

Continuous-time RL policies for robust cancer adaptive therapy.

**The question.** Does structuring a policy as a differential equation give it an
inductive bias for tumour dynamics that survives when the biology is unknown,
drifting, or from an unseen mechanism?

```
virtual patient (model_id, θ) → diffrax rollout → burden-only observation
                              → policy (MLP | ODE-RNN) → dose
```

The comparison is the point: one RL algorithm, one training loop, one evaluation
harness, with the **architecture swapped inside** and everything else held fixed.
Observation is total burden only — never subpopulations — which is what makes
this a POMDP rather than a solved control problem.

> **Status: no gated result yet.** Phases 0 and 1 have run; the Phase 2
> stop-or-go gate is registered but **unrun** (blocked on CPU, see
> [Known limitations](#known-limitations)). `LAWS.md` and `CLOSED.md` are
> deliberately empty. See [Current findings](#current-findings).

---

## Install

### Prerequisites

Only [**uv**](https://docs.astral.sh/uv/). Nothing else — not even a matching
Python.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verified with uv 0.11.3.

**You do not need to install Python 3.11 yourself.** The project requires
`>=3.11` and `.python-version` pins `3.11`; if your system Python is older, uv
downloads and manages a suitable interpreter. On the development machine the
system Python is 3.10.12 and the project runs on a uv-provisioned CPython
3.11.15 — no `pyenv`, no `conda`, no manual venv.

### Setup

```bash
git clone <this-repo> && cd Continuous_Time_RL_Policies_for_Robust_Cancer_Adaptive_Therapy
make setup          # = uv sync --group dev
```

That creates `.venv/`, resolves from the **committed `uv.lock`**, and installs
the package plus dev tooling. Exact versions are reproduced from the lockfile, so
a fresh clone gets the same JAX that produced the logged results.

Verify:

```bash
make all            # lint + test + audit + lit-check
```

Expect **51 tests passing in ~15 s**, `lit-check` OK, and `audit` reporting
`OK no hard failures` with two soft append-only WARNs (those are expected — see
[The research record](#the-research-record)).

### Running anything

Every command goes through `uv run`, which resolves the environment for you.
**There is no `activate` step and you should not need one:**

```bash
uv run python research/phase1_cohorts.py
uv run pytest -m "not slow"
```

`uv run` is not a wrapper around an activated venv — it is the supported entry
point. Activating `.venv` manually works but is unnecessary and drifts.

### Optional extras

```bash
uv sync --group dev --extra viz          # matplotlib, for Phase 1 plots
uv sync --group dev --extra baselines    # torch + gymnasium
```

`baselines` is **quarantined on purpose**. DTR-Bench/AhnChemoEnv is
PyTorch/Gym and will not `vmap`; it is an external validation anchor, never part
of the JAX simulator. Nothing under `adaptive_therapy/` may import it.

### GPU

The lockfile's default is **CPU JAX**; CUDA JAX is a locked extra:

```bash
make setup-gpu      # uv sync --group dev --extra gpu
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Verified 2026-09-23 on an RTX 4090 (driver CUDA 13.2): 51/51 tests pass, and
cohort solves agree with CPU to 5.6e-7 relative
([L166](adaptive_therapy/RESEARCH_LOG.md#L166)). Once installed, **GPU becomes
the default for every `uv run`**; force CPU per command with `JAX_PLATFORMS=cpu`.

**For this workload the GPU is slower, not faster.** Rollouts are small batches
of short sequential solves, so they are kernel-launch-bound: a 200-patient
rollout is 9× slower on GPU, a PPO update 17–40× slower. Phase 2 therefore runs
on CPU (see the note in `research/phase2_gate.py`).

---

## Layout

| Path | Role |
|---|---|
| `adaptive_therapy/envs/` | Three ODE families (Lotka-Volterra, Hahnfeldt, Kuznetsov) + PK + reward. Pure functions, no I/O. |
| `adaptive_therapy/policies/` | Actor architectures — the ablation axis. `mlp`, `ode_rnn`. |
| `adaptive_therapy/rl/` | One PPO implementation. The architecture is swapped *inside* it. |
| `adaptive_therapy/eval/` | TTP (running-nadir rule), exposure, paired Wilcoxon. |
| `adaptive_therapy/tests/` | pytest. Test names state the invariant, not the function. |
| `research/` | One-shot scripts and **all** I/O. Never imported by the package. |
| `literature/` | Paper reviews with machine-checked provenance. |
| `tools/doc_audit.py` | Record-integrity check. Hard-fails on a broken link. |

`adaptive_therapy/` is imported and tested; `research/` is scripts. Artifacts go
to gitignored `research/outputs/`.

## Reproducing the runs

All three are cheap on CPU, and **deterministic**: re-running Phase 0 and Phase 1
on this machine reproduced their committed JSON artifacts byte-for-byte, which is
what explicit seed threading (`seed: int = SEED`, never global) buys.

```bash
uv run python research/smoke_untreated.py      # ~2 s   — simulator sanity
uv run python research/phase0_identifiability.py   # ~12 s  — identifiability + separability gate
uv run python research/phase1_cohorts.py       # ~4 s   — cohort composition + untreated TTP
uv run python research/phase2_gate.py          # registered, currently blocked on CPU
```

Each writes JSON to `research/outputs/`. **The JSON artifact is the record**; the
printed summary is convenience, and log entries transcribe from the artifact,
never from an intermediate diagnostic.

## Current findings

Read [`adaptive_therapy/VERDICTS.md`](adaptive_therapy/VERDICTS.md) (30 lines),
not the log. The headlines:

- **Phase 0 gate PASSED** — the three families are separable from burden alone,
  median balanced accuracy 0.996 [0.993–1.000] over 3 seeds, 0% trajectory
  collisions. **But the margin is scale, not mechanism**: a growth-magnitude-only
  control scores 0.824, itself above the 0.80 gate. Both numbers are reported
  together, always.
- **θ is not recoverable from burden.** LV `r_R` is flat across a 49-decade
  confidence interval, independently reproducing arXiv 2608.18387 from our own
  code; Kuznetsov has 6 of 8 parameters flat. Q3 is robustness *under*
  non-identifiability — never parameter recovery.
- **PROJECT.md §12's stiffness prescription is wrong for this simulator.** The
  flagged patients are fast-transient, not stiff; fixed-step is accurate to
  ~1e-4 and adaptive `Kvaerno5` fails on them even batched alone.
- **Untreated TTP is resolution-limited** on Hahnfeldt (7.0 d) and Kuznetsov
  (14.0 d) — zero IQR at weekly sampling. LV (28.1 d [21–47]) is the only family
  with untreated dynamic range, so Phase 2 uses it.

## Known limitations

- **The Phase 2 gate is unrun.** It fails on CPU with
  `LLVM ERROR: Unable to allocate section memory` — transient fragmentation in
  LLVM's JIT allocator, **not** exhaustion: it fires with 22 GB free, and an
  identical call that failed at 16:01 succeeded at 16:04 in 4.6 s. Thread caps
  (`OMP_NUM_THREADS` and friends) do **not** help; XLA's compilation pool ignores
  them. Mitigations in place: subprocess isolation per cell, JSON shards with
  resume, bounded retry. This is an **infrastructure blocker, not a null result** —
  the registered gate stands unrun with its criteria and grid unchanged.
- **All 15 literature entries are at provenance `2°` (unread).** By
  [`literature/REVIEW_PROTOCOL.md`](literature/REVIEW_PROTOCOL.md) §0.3 a `2°`
  source **may not be cited in a design argument**, so the θ ranges in
  `envs/sampling.py` are provisional and say so in their own docstring.
- **No clinical claims.** Virtual patients only.

## The research record

This repo keeps an append-only research record, ported from two sibling repos
where the same rules ran on the honour system for months and drifted badly.
They are enforced as checks now, not discipline.

**Start a session by reading** [`LAWS.md`](LAWS.md) → [`CLOSED.md`](CLOSED.md) →
[`adaptive_therapy/VERDICTS.md`](adaptive_therapy/VERDICTS.md). Open
[`RESEARCH_LOG.md`](adaptive_therapy/RESEARCH_LOG.md) **only** at the lines a
verdict links to.

**Precedence ladder** — higher always wins, and it is global:

1. `RESEARCH_LOG.md` — append-only. Nothing overrides a logged result.
2. `LAWS.md` / `CLOSED.md` — generalizations bought with ≥2 confirmations.
3. `PROJECT.md` — the spec. Read it; **do not edit it to match a result.**
4. `ROADMAP.md`, `PREREG.md` — committed plans.
5. `CLAUDE.md` and the skills.

A lower-rung doc asserting something about a higher rung is a staleness bug *in
the lower doc*. Fix it there; never resolve it by editing upward.

**Promotion chain:** `run → log entry → VERDICTS line → CLOSED item → LAW`
(a law needs ≥2 independent confirmations). Entries are capped at **40 lines,
hard** — if an entry will not fit, the run needed an artifact, not more prose.

Before any run that could produce a gateable result, register it in
`PREREG.md` with exact pass numbers, kill criteria, and a declared search space:

> If you cannot write a kill criterion, the experiment is not ready to run.

`make audit` hard-fails (exit 1) on a dangling `RESEARCH_LOG.md#Lnnnn` link or a
stale generated file. It warns on append-only deletions, index coverage and the
line budget — deliberately soft, because git history is immutable and a
permanently-red check trains everyone to ignore the tool.

`literature/PAPERS.md` is **generated**. Never hand-edit it:

```bash
uv run python research/lit_table.py --write
make lit-check      # exit 1 if stale
```

## Non-negotiables

- **Burden-only observation.** Never subpopulations.
- **Report exposure with TTP, always.** Winning TTP by dosing constantly is
  aggression, not robustness.
- **Medians + IQR, never means** — trajectories are heavy-tailed.
- **Paired Wilcoxon** across matched virtual patients, never a t-test.
- **Multiple seeds** with variance bands. Single-seed is a desk-reject.
- **Strict 2×2** for architecture × curriculum. No confounded comparisons.
- **No citation without a resolving DOI/venue**; a `2°` source may not be cited
  in a design argument.
- **No clinical-outcome claims.** Virtual patients only.

## Development

```bash
make lint      # ruff check + format --check
make fmt       # apply fixes
make test      # pytest -m "not slow"
make audit     # record integrity — exit 1 on a broken link
make lit-check # literature index freshness
make all       # all of the above
```

Install the hooks once with `uv run pre-commit install`; they run the same
checks as CI.

Commits are tagged `[research]`, `[feat]`, `[fix]`, `[chore]`, `[docs]`, and the
subject states the **finding**, not the file list. Stage only the result's own
files — never `git add -A`.

## Further reading

- [`PROJECT.md`](PROJECT.md) — the full spec, including PART VII on how this
  project works.
- [`ROADMAP.md`](ROADMAP.md) — phases, kill criteria, the traps this program must
  not fall into, and the honest prior.
- [`literature/REVIEW_PROTOCOL.md`](literature/REVIEW_PROTOCOL.md) — provenance
  tiers and the citation rule.

[L88]: adaptive_therapy/RESEARCH_LOG.md#L88
