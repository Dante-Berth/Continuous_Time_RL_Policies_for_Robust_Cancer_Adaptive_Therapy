# PREREG — adaptive_therapy

Hypotheses and their kill criteria, written BEFORE the run that tests them.
Append-only; a mid-run change is a dated AMENDMENT line, never an edit.

See [.claude/skills/prereg/SKILL.md](../.claude/skills/prereg/SKILL.md).
`/logentry` checks each result against its registration here.

**If you cannot write a kill criterion, the experiment is not ready to run.**

---

## [2026-09-22] — Untreated-trajectory smoke run across all three families

**Hypothesis:** the three implemented families produce numerically sound,
mechanistically distinct untreated trajectories from the literature parameter
sets — so the simulator is fit to build Phase 1's cohort sampler on. This is an
infrastructure check, not a research claim: it tests the code, not the biology.

**Success criteria:** for each of the three families, over 365 days untreated:
(1) all states finite, no NaN/Inf; (2) burden strictly positive throughout;
(3) each family's trajectory shape is distinguishable from the other two by
its turning-point count or its dynamic range.

**Kill criteria:** any NaN, any negative burden, or two families producing
indistinguishable trajectory shapes ⇒ stop and fix the simulator before writing
the cohort sampler. Indistinguishable families would mean the Tier-2 mechanism
shift is not a real shift, which is a Phase-0 finding, not a smoke-test pass.

**Cohort/plane:** one representative θ per family (the literature values in
`tests/test_envs.py`), untreated (u=0), t ∈ [0, 365] d, fixed-step Tsit5.
n = 1 per family — this is a shape check, not a cohort statistic.

**Declared search space:** none. No hyperparameter is varied and none may be
added; a smoke run that needs tuning to pass is a failing smoke run.

**Planned artifacts:** `research/smoke_untreated.py`,
`research/outputs/smoke_untreated.json`.

**Not a gateable research result.** No TTP, no exposure, no seeds, no Wilcoxon —
none apply to a deterministic single-patient shape check. It is registered
because it is the first run this record ever sees, and the protocol should be
exercised on something harmless before it matters.
