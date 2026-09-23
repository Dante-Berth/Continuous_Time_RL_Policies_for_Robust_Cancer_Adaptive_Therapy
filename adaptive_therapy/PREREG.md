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

## [2026-09-22] — Phase 2 STOP-OR-GO: does a continuous-time policy beat a discrete one on LV under a Tier-1 shift, at matched exposure?

**Hypothesis.** A Neural-ODE policy carries an inductive bias for continuous
tumour dynamics that a discrete MLP lacks, so under a parameter shift outside
the training range it should schedule *better per unit of drug* — longer TTP at
the same cumulative exposure. Mechanism, not vibes: the tumour state evolves
continuously between monitoring visits, and an ODE-structured hidden state
integrates that interval explicitly rather than treating it as one opaque step.

**Why LV, and why exposure-matched.** LV is the only family with untreated
dynamic range (Phase 1: 28.1 d [21–47], 29% censored; Hahnfeldt 7.0 d and
Kuznetsov 14.0 d at zero IQR). And TTP rises monotonically with dose on LV
(u=0→28 d, 0.25→42, 0.5→74, 0.75→246, 1.0→295 d), so a pure-TTP gate is won by
constant maximum dosing and measures aggression, not skill.

**Success criteria — ALL must hold.**
1. **Exposure-matched TTP:** median TTP(ODE) − median TTP(MLP) > **0** at
   matched median cumulative exposure (within ±10%), on the **Tier-1 shifted**
   cohort.
2. **Paired Wilcoxon p < 0.05** across matched virtual patients (same θ, both
   architectures).
3. Holds at **≥ 3 seeds**, reported as median + range, and at **both** exposure
   penalty weights.
4. Both arms must beat the **untreated** control on TTP — an arm that does not
   is not a policy, and the comparison between two non-policies is meaningless.

**Kill criteria — no re-parametrization, no grid widening.**
- **Criterion 1 fails (ODE ≤ MLP at matched exposure)** ⇒ **STOP. Write the
  negative result.** The multi-mechanism curriculum does not rescue an
  architecture that loses on one clean family. Do NOT widen the network, add
  training steps, tune the learning rate, or try a third architecture.
- **Criterion 2 fails (p ≥ 0.05) with criterion 1 passing** ⇒ the effect is not
  distinguishable from noise at this cohort size. Report as a null; do not
  enlarge the cohort to chase significance.
- **Either arm fails to beat untreated** ⇒ a training bug, not a result. Fix
  and re-run; the re-run is registered as an AMENDMENT.

**Cohort/plane.** LV only. Train on θ in the `sampling.py` ranges; **Tier-1
test** on θ drawn outside them (r_S and r_R scaled ×1.5, K ×0.6 — fixed here,
before any run). 200 train / 200 test patients, burden-only observation with
σ=0.05 multiplicative noise, weekly decisions over 365 d, dose ∈ {0, ½, 1}.

**Declared search space.** Architecture ∈ {MLP, Neural-ODE} × exposure weight ∈
{0.05, 0.20} × seed ∈ {0,1,2} = 12 runs. Hidden width 64 and learning rate 3e-4
for BOTH arms, fixed, not tuned. Parameter counts reported; if they differ by
more than 2× the comparison is confounded and must be re-run matched.

**Standing criteria.** Median + IQR never means; cumulative exposure reported
with every TTP; burden-integral as the second endpoint; paired Wilcoxon; ≥3
seeds. TTP by the running-nadir rule, with `from_baseline` alongside.

**Planned artifacts.** `adaptive_therapy/policies/{mlp,ode_rnn}.py`,
`adaptive_therapy/rl/ppo.py`, `research/phase2_gate.py`,
`research/outputs/phase2_gate.json`, `research/outputs/phase2_gate.png`.

**Honest prior.** The edge is expected to be SMALL or ABSENT here. Tier-1 is a
parameter shift, not a mechanism shift, and the continuous-time advantage has a
mechanical reason to appear under *irregular sampling* (Tier 4) rather than
under regular weekly visits. A null on this gate is a publishable result per
PROJECT.md §15 and is the registered, expected outcome.

**AMENDMENT [2026-09-22, before any run]:** the registration declared hidden
width 64 for BOTH arms and required parameter counts within 2×. Building them
showed that gives MLP 516 vs ODE-RNN 3076 — a 5.96× gap, which my own criterion
rejects as confounded. **MLP width is therefore 384, giving exactly 3076
parameters (ratio 1.00); ODE-RNN stays at width 64, hidden 16.** Declared before
any training run, so this is a specification fix, not a selection event: no
result has been seen. The ODE arm is now the *smaller-width* network, and a win
for it cannot be attributed to capacity.

## [2026-09-23] — GPU re-measurement of the §12 stiffness finding (L88), paired against CPU in one process

**Hypothesis:** L88's finding (fixed-step accurate on fast-transient Hahnfeldt
patients; adaptive `Kvaerno5`+PID fails on them) is a property of the ODE and
the controller, not of CPU-XLA SIMD, so it transfers to GPU. **Rival mechanism,
tested here:** L88 ran `rtol=1e-6` in **float32** (eps ≈ 1.2e-7); roundoff
may be what makes the controller thrash, not the patients.
**Success criteria (finding transfers) — ALL on GPU:** (i) fixed-step
dt0=0.05 vs dt0=0.005 on the flagged patients, max relative burden error
≤ 1e-3; (ii) `Kvaerno5`+PID at L88's tolerances (1e-6/1e-8, max_steps 200k)
fails (exception or non-finite) on the flagged batch alone; (iii) CPU↔GPU
fixed-step burden parity ≤ 1e-4 max relative.
**Kill / reversal:** (ii) passes on GPU with results within 1e-3 of fine
fixed-step ⇒ the finding is backend-specific; log it, fixed-step stays default
only if also faster. (iii) fails ⇒ backends are not interchangeable: Phase 2
runs entirely on one backend, and CPU JSONs are never mixed into a GPU comparison.
**Diagnostic (not a gate):** adaptive at loose tolerance (1e-4/1e-6), both
backends. Succeeds where 1e-6 fails ⇒ L88's failure is a float32 precision
floor, and gate.py's comment must be corrected in a new entry (not by editing L88).
**Cohort:** Hahnfeldt, 200 LHS, seed 0, Phase 1's CFG (dt0=0.05, 53 saves);
flagged = Phase 1's fast-transient rule. Timing (fixed-step n=200, median of 5,
post-compile) reported per backend, not gated. No seeds: deterministic solve.
**Planned artifacts:** `research/gpu_stiffness.py`, `research/outputs/gpu_stiffness.json`.

**AMENDMENT [2026-09-23, AFTER seeing the registered results — selection event,
declared]:** registered criteria (i)–(iii) all passed, unchanged. A post-run
3-patient check showed adaptive *explicit* Tsit5+PID solving at L88's tolerances
in ~500 steps, and Kvaerno5 also failing on an *unflagged* patient — contradicting
L88's mechanism (fast transient ⇒ adaptive thrash). Added **diagnostic arms, not
gates**: per-patient success (throw=False) of Kvaerno5 and Tsit5 adaptive at both
tolerances over ALL kept patients, split flagged vs bulk, CPU only (backend parity
already passed). Declared post hoc; it may revise L88's *mechanism*, never
Phase 2's criteria.
