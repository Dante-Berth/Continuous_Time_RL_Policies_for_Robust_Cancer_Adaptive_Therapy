# Doc maintenance — always applies

## The research record

The past/present/future split is enforced by `tools/doc_audit.py`. Run it before
committing any doc change:

    make audit          # or: uv run python tools/doc_audit.py -v

- **`adaptive_therapy/RESEARCH_LOG.md` is append-only, forever.** ≤ 40 lines per
  entry is a hard cap. Never edit an existing entry — it invalidates every
  `#Lnnnn` link below the edit.
- **After appending an entry, update `VERDICTS.md` in the same commit.** An
  entry without an index update is invisible to future sessions.
- Program-level closures are promoted to root `CLOSED.md` with a re-open
  condition. A pattern confirmed by **≥ 2 closures** is promoted to `LAWS.md`.
- **Killed work stays described**, struck through with a link to the entry that
  killed it — so a future session sees it was considered and why it was not
  built.

## The precedence ladder

Stated once, in [CLAUDE.md](../../CLAUDE.md), and global. A lower-rung doc
asserting something about a higher rung is a **staleness bug in the lower doc**.
Fix it there; never resolve it by editing upward.

**`PROJECT.md` is the spec. Read it; do not edit it to match a result.** A
result that contradicts the spec is a log entry, then a verdict — not a quiet
edit to the spec.

## Generated documents

`literature/PAPERS.md` is generated from `literature/papers/*.md` front-matter.
**Never hand-edit it** — edit the source and regenerate:

    uv run python research/lit_table.py --write
    make lit-check      # exit 1 if stale

Any document that mixes intent with live state gets the same treatment: status
prose is **computed, never typed**. This is what stops a file accreting
"as of <date>, …" paragraphs nobody updates.

## Literature

A paper is not evidence for this program; only a logged, gated run is.
Literature sits **below everything** on the ladder. A `2°` (unread) source may
not be cited in a design argument — see
[literature/REVIEW_PROTOCOL.md](../../literature/REVIEW_PROTOCOL.md) §0.3.
