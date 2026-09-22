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

## [2026-09-22] — Phase 0 gate: families separable under burden-only, but on scale not mechanism; θ recovery fails as predicted
**Verdict:** OPEN — gate PASSES, so Q3 needs no reframe; the separability margin is weaker than the headline · **Gate:** median balanced accuracy 0.996 [0.993–1.000] over 3 seeds vs ≥0.80 registered; 0% collisions on all 3 pairs; TTP/exposure/Wilcoxon n-a (no policy trained) · **Artifacts:** `research/phase0_identifiability.py`, `research/outputs/phase0_identifiability.json`

### Context
PROJECT.md §5's day-1 gate: if the families are not separable under burden-only,
a mixture encoder cannot separate them and Q3 must be reframed before any
encoder is written. Registered with the 365 d window fixed in advance.

### What was tried
900 LHS patients (300/family), untreated, 52 weekly samples, multiplicative
lognormal noise σ=0.05. 1-NN on normalised log-burden **shape** (first sample
subtracted, max-normalised) — deliberately a near-lower-bound extractor, so a
pass means the information is trivially present, not that a clever model found
it. Identifiability by reverse-mode AD through the solver: FIM in
d(log burden)/d(log θ), per-component profile CI.

### Results
**Met as registered — PASS**, and separately **weaker than it looks**.

A post-hoc control (declared as an AMENDMENT/selection event in PREREG.md):
1-NN on the **single scalar** log B(365) − log B(0) scores **0.824** — already
above the registered gate on its own. Full 52-point shape adds only +0.17. The
families differ in gross growth magnitude (log-growth LV +0.39, Hahnfeldt +4.26,
Kuznetsov +3.24±6.40), so **the gate passes on scale, not on mechanism**.

θ recovery fails as expected (per-component classes in the artifact). LV
**`r_R` flat at a 49-decade CI, sensitivity 2.1e-3** — independently reproducing
arXiv 2608.18387's weak-identifiability proof from our own code; `d_R` likewise,
both acting only on the resistant ~1% of a summed observable. Kuznetsov worst:
**6 of 8 flat**. Bug fixed mid-run: drug params have exactly zero sensitivity
under u=0 and the pseudo-inverse read that as *perfectly* identifiable; now
classed `no-effect`.

### Decision
Proceed to Phase 1. Q3 is **not** reframed — the gate passed. Three constraints
carried forward: (1) any future separability claim must report the growth-only
control beside it; (2) Tier-2 on untreated dynamics is an easy test — the real
one is under treatment, where mechanisms diverge beyond scale; (3) Kuznetsov's
6/8 flat directions mean an encoder targeting its θ is near-hopeless, so the
MoE should be assessed on cluster purity, never parameter recovery.
