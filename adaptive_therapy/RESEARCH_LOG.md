# RESEARCH LOG — adaptive_therapy

Append-only, forever. ≤ 40 lines per entry (hard cap). Newest entries at the
bottom. See root [CLAUDE.md](../CLAUDE.md) for the entry format and the
`/logentry` protocol.

Rung 1 of the precedence ladder: a logged, gated result beats every other
document in this repo, including [PROJECT.md](../PROJECT.md). Nothing here is
ever edited — a correction is a new entry that links back.

---

## [2026-09-22] — Simulator smoke: three families produce sound, distinguishable untreated trajectories

**Verdict:** OPEN — the simulator is fit to build the Phase-1 cohort sampler on · **Gate:** n-a (infrastructure check: no TTP/exposure/seeds/Wilcoxon apply to a deterministic n=1 shape check) · **Artifacts:** `research/smoke_untreated.py`, `research/outputs/smoke_untreated.json`

### Context
First run this record has seen. Registered in PREREG.md [2026-09-22] **before**
running, primarily to exercise `/prereg` → run → `/logentry` on something
harmless before the loop has to carry a result that matters.

### What was tried
One representative θ per family from the literature values, untreated (u=0),
t ∈ [0,365] d, fixed-step Tsit5, n=1 per family. No hyperparameter varied
(declared search space: none).

### Results
**Met as registered — PASS on all three criteria.** Per-family numbers are in
the artifact; shapes (log10 range, turns): LV 0.3/0, Hahnfeldt 2.2/1,
Kuznetsov 2.7/0. All finite, all strictly positive, no two families sharing
both. LV saturates at K as competition predicts; Hahnfeldt turns as vasculature
re-equilibrates; Kuznetsov climbs ~500× under immune predation.

Separately verified in `tests/test_envs.py`: under full dose the LV sensitive
population falls 7588→3222 while resistant **rises** 287→315 — competitive
release, the mechanism adaptive therapy exploits, reproduced.

### Decision
Proceed to Phase 1 (cohort sampler + untreated sanity gate). Two constraints
carried forward, both already visible here:
1. Kuznetsov at 365 d shows 0 turning points but oscillates over 1000 d
   (≥2 turns, burden → ~1e-7 dormancy then recovery). The **horizon changes
   the apparent dynamics**, so the sanity gate's observation window is a
   modelling choice, not a detail.
2. This says nothing about `vmap` cost — n=1 per family. The PROJECT.md §12
   stiffness trap remains untested.
