---
name: prereg
description: Pre-register an experiment BEFORE running it — hypothesis, exact success/kill criteria, declared search space, and planned artifacts, appended to adaptive_therapy/PREREG.md. Run at the start of any session that will produce a gateable result; /logentry checks the result against this entry.
---

# /prereg — pre-registration protocol

## Step 0 — Check it isn't already dead

Before registering anything, check the idea against root
[CLOSED.md](../../../CLOSED.md), [LAWS.md](../../../LAWS.md), and the CLOSED
section of [VERDICTS.md](../../../adaptive_therapy/VERDICTS.md). If it matches a
closed item and the stated re-open condition is not met, **stop here** — do not
register, do not run.

Re-opening requires new data or a new mechanism, never a re-parametrization.

## Step 1 — Append the registration

Append to `adaptive_therapy/PREREG.md` (append-only, never edit past entries):

    ## [YYYY-MM-DD] — One-line experiment name
    **Hypothesis:** what specific effect exists and why (mechanism, not vibes).
    **Success criteria:** the EXACT numbers that count as a pass
      (e.g. median TTP improvement > X days, paired Wilcoxon p < 0.05, over >= N seeds).
    **Kill criteria:** what result means STOP — no re-parametrization, no grid widening.
    **Cohort/plane:** which families, which theta ranges, which tier, how many patients.
    **Declared search space:** every hyperparameter varied and its values. No widening.
    **Planned artifacts:** script path, output JSON path.

Target ≤ 15 lines.

> **If you cannot write a kill criterion, the experiment is not ready to run.**

### This project's standing criteria (from the spec)

Any registration touching a policy comparison must state:

- **median TTP + IQR**, never a mean (trajectories are heavy-tailed);
- **cumulative drug exposure**, always alongside TTP — a policy that wins TTP by
  dosing constantly is aggressive, not robust;
- a **second endpoint** (metastasis-proxy / burden-integral), because TTP and
  metastasis risk pull against each other;
- **paired Wilcoxon signed-rank** across matched virtual patients, never a t-test;
- **the seed count**, ≥ 3. Single-seed is a desk-reject, not a result.

## Step 2 — Run the experiment

Only after Step 1 is written to disk. Any mid-run change to criteria, grid, or
cohort must be appended as a dated **AMENDMENT** line under the entry — never
edited in place. An amendment after seeing partial results is a **selection
event** and must be declared in the log entry's evidence.

## Step 3 — Close the loop

At session end, `/logentry` quotes this registration and states one of:
**met as registered** / **failed as registered** / **criteria amended mid-run
(selection risk)**. A result with no prereg entry is still loggable — but the
entry must say "not pre-registered" explicitly.
