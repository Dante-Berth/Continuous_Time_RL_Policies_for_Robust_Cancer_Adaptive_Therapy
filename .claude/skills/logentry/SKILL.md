---
name: logentry
description: Session-end research-log protocol — append the RESEARCH_LOG.md entry, update VERDICTS.md in the same commit, and promote program-level closures to root CLOSED.md. Run at the end of any session that produced a result or ruled out a direction.
---

# /logentry — session-end log + index protocol

/ prereg opens a run; this closes it. Run it at the end of any session that
produced a result **or ruled out a direction** — a dead end that never got
logged is one the program will re-propose in three months.

## Step 1 — Gather (from THIS session, not from memory)

- the one-line summary of what was found or ruled out;
- verdict status: **OPEN** (result, question not settled) / **FROZEN** (settled
  positive) / **CLOSED** (ruled out — include the do-not-re-litigate reason);
- gate evidence: median TTP + IQR, cumulative exposure, second endpoint,
  Wilcoxon p, seed count — **transcribed from the final completed run's
  artifact**, never from an intermediate diagnostic;
- artifact paths.

If a metric cannot be traced to a concrete run output, **say so** rather than
writing a number.

Check `adaptive_therapy/PREREG.md` and state one of: **met as registered** /
**failed as registered** / **criteria amended mid-run (selection risk)**. If
there is no registration, write "not pre-registered" explicitly.

## Step 2 — Append the log entry

Append to `adaptive_therapy/RESEARCH_LOG.md`. **Never edit an existing entry.**

    ## [YYYY-MM-DD] — One-line summary
    **Verdict:** OPEN|FROZEN|CLOSED — half-line takeaway · **Gate:** median TTP x.x [IQR a-b] / exposure y / 2nd endpoint z / Wilcoxon p=0.0x / n seeds · **Artifacts:** paths
    ### Context
    ### What was tried
    ### Results
    ### Decision

**≤ 40 lines is a hard cap, not a target.** If the entry will not fit, the run
needed an artifact, not more prose.

Write the header to be greppable: a reader searching for "Tier 2" should find
entries *about* Tier 2, not every entry that mentions it in passing.

## Step 3 — Update VERDICTS.md in the SAME commit

    - **claim in bold — headline number** → [Lnnnn](RESEARCH_LOG.md#Lnnnn)

where `Lnnnn` comes from:

    grep -n "^## \[" adaptive_therapy/RESEARCH_LOG.md | tail -1

The log is the archive; VERDICTS is the index. **An entry without an index
update is invisible to future sessions.** If this result settles an existing
OPEN line, **move** it to CLOSED rather than duplicating. Never delete a CLOSED
line.

## Step 4 — Promote program-level closures

If the result closes a direction for the whole program (not just one phase), add
a numbered item to root `CLOSED.md` with its **re-open condition** stated inline.

If a pattern now has **two independent confirmations** across separate closures,
propose a law in `LAWS.md` — one result is a verdict, not a law.

## Step 5 — Consistency check

    grep -c "^## \[" adaptive_therapy/RESEARCH_LOG.md      # entry count grew by 1
    grep -A1 "^## \[" adaptive_therapy/RESEARCH_LOG.md | tail -4   # Verdict line present
    make audit                                             # links resolve, exit 0

## Step 6 — Report

One short paragraph, including any number that could not be verified.

## Step 7 — Commit (standing authorization)

- Stage **only** this result's own files. Never `git add -A`.
- Message: `[research] <one-line result>`, body carrying the mechanism and the
  gate outcome.
- **Do NOT push.**
- If the result changes a conclusion that other work depends on, stop and ask.
