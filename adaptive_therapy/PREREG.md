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

## [2026-09-22] — Phase 0: practical identifiability of θ and separability of the three families from burden-only

**Hypothesis.** Two claims, deliberately separated because they have different
consequences. (a) **Identifiability:** θ is only *weakly* identifiable from the
scalar burden — some directions in θ-space are recoverable, others are flat,
and the flat directions are why a discrete policy that implicitly estimates θ
should fail under shift. This is expected to FAIL in the recovery direction;
arXiv 2608.18387 proves `r_R` is weakly identifiable on this same LV model, so
a result showing clean recovery would indicate a bug in our Fisher/profile code,
not a discovery. (b) **Separability:** despite (a), the three families produce
burden trajectories distinguishable *from each other*. Separability is the
weaker and more useful property — Tier-2 and any future mixture encoder depend
on it, and it can hold even where θ-recovery fails.

**Success criteria.**
1. **Identifiability, per family:** report the Fisher information matrix
   eigenspectrum and the profile likelihood for every θ component. A component
   is `identifiable` if its 95% profile-likelihood CI is bounded and spans
   < 1 order of magnitude; `weak` if bounded but wider; `flat` if unbounded
   within the sampled range. **No pass/fail threshold is attached to this** — it
   is a measurement, and its value is the classification itself.
2. **Separability (this is the gate):** a nearest-neighbour classifier on
   log-burden trajectories alone must identify the generating family with
   **balanced accuracy ≥ 0.80** over a held-out third of the cohort, with the
   confusion matrix reported. Chance is 0.33.
3. **Trajectory collision:** report the fraction of cross-family trajectory
   pairs whose burden curves agree within measurement noise (relative L2
   < 5%) over the full window. Report per family-pair, not pooled.

**Kill criteria — no re-parametrization, no grid widening.**
- **Balanced accuracy < 0.80** ⇒ the families are NOT separable under
  burden-only. **Stop, and reframe Q3 before writing any encoder** (PROJECT.md
  §5). Do not rescue by lengthening the window, widening θ ranges, or adding an
  observable — each of those changes the research question rather than answering
  it. A reframe is the registered outcome here, not a failure of the run.
- **Balanced accuracy ≥ 0.80 but one family-pair collides > 20%** ⇒ that pair is
  the hard case; Tier-2 must be reported per-pair rather than pooled, and the
  claim "mechanism shift" is restricted to the separable pairs.
- Any NaN in a profile likelihood that is not traceable to a solver failure ⇒
  fix the code before reading any number from this run.

**Cohort/plane.** All three families. n = 300 per family (900 total) via Latin
hypercube over the per-family θ ranges in `envs/*.py`, untreated (u=0), the
observation window fixed at **t ∈ [0, 365] d** with 52 weekly samples —
clinically plausible monitoring, and fixed BEFORE seeing any result. Burden
observed with multiplicative lognormal noise σ=0.05. Patients failing the
untreated sanity gate are logged and excluded, with the rejection count reported.

**Declared search window — the one real degree of freedom.** The 365 d window is
declared here and **may not be changed after seeing the accuracy**. The smoke run
already showed Kuznetsov looks monotone at 365 d and oscillates by 1000 d, so a
longer window would likely raise separability — which is exactly why extending it
post-hoc would be a selection event. If a second window is ever run, it is a
NEW registration reported alongside this one, never a replacement.

**Seeds.** 3 LHS seeds (0, 1, 2); report balanced accuracy as median + range
across seeds, not a single number.

**Standing policy criteria: N/A.** No policy is trained here — no TTP, exposure,
second endpoint or Wilcoxon applies. This is a property of the simulator, not of
a controller.

**Planned artifacts.** `research/phase0_identifiability.py`,
`research/outputs/phase0_identifiability.json`,
`research/outputs/phase0_separability.png`.

**AMENDMENT [2026-09-22, post-hoc]:** after seeing balanced accuracy 0.996, a
growth-magnitude-only control was added that was NOT registered — a 1-NN on the
single scalar `log B(365) − log B(0)`. It scores **0.824**, itself above the
registered 0.80 gate. This is declared as a **selection event**: it was run
because the headline looked too easy, and it materially weakens the
interpretation (the gate passes on scale, not on mechanism). It does not change
the registered pass/fail outcome, which stands as recorded. Any future
separability claim must report this control alongside the headline.
