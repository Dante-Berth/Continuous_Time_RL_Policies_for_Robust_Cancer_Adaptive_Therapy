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

**AMENDMENT [2026-09-23, AFTER run 1 completed — selection event, declared]:**
run 1 ran 12/12 and printed FAIL, but its Tier-1 test cohort was invalid (see
RESEARCH_LOG, run-1 entry): y0 was built from the *unshifted* K, putting 43–50%
of test patients above carrying capacity — they shrink untreated, and 14–18%
diverge to NaN under full dose, which the TTP rule scored as *censored*. Per
this entry's "bug ⇒ fix and re-run as an AMENDMENT" clause: (1) y0 re-derived
from the shifted K, keeping each patient's sampled f_R and n_0; (2) non-finite
burden now raises instead of scoring; (3) PASS now also requires the ±10%
exposure match that criterion 1 always stated. **Unchanged:** hypothesis,
criteria, shift multipliers, grid, seeds, width, LR, n, and the estimator
(median over progressors). **Added secondary:** pooled censoring rate per arm.
**Run 2 is final.** Its verdict stands whatever it shows; a run 3 is justified
only by a crash or a non-finite rollout, and would void run 2, never re-grade it.

## [2026-09-23] — Phase 2b STOP-OR-GO: Phase 2 re-registered on a corrected trainer, with a closed-loop validity gate

**Why a new registration.** Phase 2 FAILED as registered ([L275](RESEARCH_LOG.md#L275)) but compared a
feedback MLP with an open-loop ODE-RNN ([L312](RESEARCH_LOG.md#L312)). Diagnosis ([L353](RESEARCH_LOG.md#L353)): the default-init
ODE-RNN saturates to a deterministic schedule within ~10 visits *before
training*; the entropy bonus had zero expected gradient; the ODE critic read a
stale state; PPO re-sampled actions and left advantages attached. All fixed for
BOTH arms (trainer: stored-action recurrent PPO, exact entropy, post-observation
critic; ODE latent-field output init ×0.1). Phase 2's FAIL stands — this is a
new test, never a re-grade. Written BEFORE the fixed-trainer diagnostic finished.
**Hypothesis:** unchanged from Phase 2 [2026-09-22].
**Criteria — ALL must hold:** (0) **validity:** every cell of both arms has
`observation_dependence` ≥ **0.10** on its Tier-1 test rollouts — if any cell
fails, the run is a TRAINING FAILURE with no verdict, and the next step is a
fresh registration with another algorithm (SAC-discrete / recurrent TD3); (1)
median TTP(ODE) − median TTP(MLP) > 0 with median exposure within ±10%; (2)
**one-sided** paired Wilcoxon, ODE later than MLP, p < 0.025 — Phase 2's
two-sided p could be "significant" in MLP's favour; (3) both weights, 3 seeds;
(4) both arms' median TTP > untreated.
**Kill:** (1) fails ⇒ STOP, negative result. (2) fails with (1) passing ⇒ null;
no cohort enlargement.
**Cohort / search space:** identical to Phase 2 after its amendments (LV,
Tier-1 ×1.5/×1.5/×0.6 with y0 from shifted K, 200/200, σ=0.05, weekly, 52 visits;
{MLP, ODE-RNN} × w {0.05, 0.20} × seeds {0,1,2}; 3076 params each; LR 3e-4,
300 updates, entropy_coef 0.01) — fixed, not tuned.
**Secondary:** pooled censoring, burden-integral, progressor-only-median caveat.
**Declared risk:** exposure matching failed in both Phase 2 runs because each arm
picks its own dose; (1) may fail for reasons unrelated to architecture. Left
unchanged — relaxing it after Phase 2 would tailor the criterion to the data.
**Artifacts:** `research/phase2_gate.py`, `research/outputs/phase2b_gate.json`,
trained policies in `research/outputs/phase2b_policies/`.
**Honest prior:** a null on Tier 1 remains the expected outcome.

## [2026-09-23] — Phase 2c STOP-OR-GO: Phase 2 re-registered on PQN for both arms, with a censoring-correct TTP median

**Why.** Phase 2b was INVALID ([L391](RESEARCH_LOG.md#L391)): under corrected
PPO, 3/6 ODE-RNN cells still ended open-loop; MLP 6/6 closed-loop. Per 2b's own
kill clause the next step is another algorithm for BOTH arms. PQN
(arxiv-2407.04811, reviewed 1°) needs no replay buffer, so the recurrent arm needs
no stored hidden states, and its ε floor cannot collapse like a learned entropy.
Registered BEFORE any PQN run on the gate cohorts (only a 12-update smoke test
on 64 in-range patients, which computes no TTP). Phases 2 and 2b are not re-graded.
**Hypothesis:** unchanged from Phase 2 [2026-09-22].
**Trainer (fixed, not tuned):** `rl/pqn.py` defaults = Table 4 (p.36): LR 3e-4
linear decay, RAdam, grad-clip 0.5, γ 0.99, λ 0.5, ε 1.0 → 0.005 over the first
10% of updates, 4 epochs × 4 minibatches of WHOLE episodes; 300 updates × 32
patients (the PPO budget); reward scale 0.01. **Arms:** Q-networks with
LayerNorm on the readout — MLP width 441 (3090 Q-path params) vs ODE-RNN (3091);
critic heads unused. Greedy (argmax) evaluation.
**Criteria — ALL must hold:** (0) **validity:** every cell has greedy
`observation_dependence` ≥ **0.10** (fraction of patient-visits whose greedy dose
changes when the observation history is swapped); any cell below ⇒ TRAINING
FAILURE, no verdict. (1) **capped** median TTP (min(TTP, horizon), all
patients) ODE ≥ MLP — a guard that the medians do not contradict (2), not the
evidence of superiority; median exposure within ±10%. (2) **one-sided** paired
Wilcoxon, ODE later than MLP, p < 0.025 — this carries the direction. (3) both
weights, 3 seeds. (4) both arms' capped median > untreated capped median.
**Kill:** (0) holds and (2) fails ⇒ STOP, negative result, written as "no
continuous-time edge with either trainer". (1)'s exposure match fails with (2)
passing ⇒ report as exposure-confounded, not a pass.
**Cohort / grid:** identical to Phase 2b (LV, Tier-1 ×1.5/×1.5/×0.6 with y0 from
shifted K, 200/200, σ 0.05, weekly × 52; {MLP, ODE-RNN} × w {0.05, 0.20} ×
seeds {0,1,2}).
**Declared asymmetries / risks:** PQN's stability theorem is a wide-critic
result (p.5–6): it plausibly covers the 441-wide MLP layer better than the
16-wide ODE state. ε-greedy is weak exploration (p.9). Table 4 was tuned on
Craftax, not here. Exposure matching failed in 3 of 4 Phase-2 comparisons
because each arm picks its own dose.
**Artifacts:** `research/phase2_gate.py phase2c`, `research/outputs/phase2c_gate.json`,
`research/outputs/phase2c_policies/`.
**Honest prior:** a null on Tier 1 remains the expected outcome; a VALID run is
itself informative, since no previous run managed one.

## [2026-09-24] — W3: memory vs continuous time on Tiers 0 and 1, compared on TTP-vs-exposure frontiers (W2's rule)

**Why.** Phase 2c compared a memoryless MLP with an ODE-RNN, so it could not say
whether the loss came from memory or from continuous time
([L424](RESEARCH_LOG.md#L424)). W1 then found the ODE-RNN worse in-distribution
too, making L424 a training/capacity gap, not a robustness gap
([L461](RESEARCH_LOG.md#L461)). Exposure was matched ±10% in only 1 of 8 Phase-2
comparisons, since each arm picks its own dose. Registered BEFORE the LSTM arm
is built and before any run of this grid.
**Hypotheses.** H_mem: an LSTM's frontier lies above the MLP's. H_ct: the
ODE-RNN's frontier lies above the LSTM's — the continuous-time question.
**Arms (PQN, Phase-2c trainer, fixed, not tuned):** MLP (width 441, 3090),
ODE-RNN (3091), **LSTM** — `eqx.nn.LSTMCell(1, 26)` + the same LayerNorm Q
readout, ~3045 Q-path params; the exact count, within ±5% of 3090, is asserted
by a test before any run. Same observation, `dt` signature and greedy eval.
**Grid:** 3 arms × w ∈ {0.0, 0.05, 0.10, 0.20, 0.40} × seeds {0, 1, 2} = 45
cells. Every cell is evaluated on **Tier 0** (in-range, the Tier-1 patients
before the shift) and **Tier 1** (×1.5 r_S, ×1.5 r_R, ×0.6 K, y0 from the
shifted K), 200 test patients each, LV, σ 0.05, weekly × 52.
**Frontier rule (W2, fixed here).** Per arm and tier, at each w: take the median
over seeds of the cell's median cumulative exposure (x) and capped median TTP
(y). The arm's frontier is the piecewise-linear curve through its 5 points,
sorted by x. **Gap(A, B)** = mean of y_A − y_B over the exposure overlap of
the two curves (100-point grid), in days. **CI:** 2000 bootstrap resamples of the
200 test patients (the same indices for every cell, preserving pairing),
recomputing every cell median and the gap; 95% percentile interval.
**Criteria.** (0) **validity:** each cell's greedy `observation_dependence`
≥ 0.10; an arm failing in any cell gets a trainability result and no frontier
verdict. (0b) The overlap must span ≥ 20% of both arms' exposure ranges, else
"frontiers do not overlap" is the result, not a gap. **H_ct passes** iff
Gap(ODE, LSTM) on Tier 1 > 0 with CI excluding 0. **H_mem**: the same for
Gap(LSTM, MLP). **Robustness wording** (vs. "better controller") is allowed only
if the Tier-1 gap CI lies above the Tier-0 gap point estimate.
**Kill.** H_ct fails ⇒ "no continuous-time edge over a matched recurrent arm on
Tiers 0/1", logged; W4 proceeds, since it is a separate hypothesis. No arm
widened, no retuning, no grid extension after seeing results.
**Secondary (descriptive):** pooled paired one-sided Wilcoxon per w as in 2c;
censoring; per-seed points on the frontier plot; obs-explained dose fraction (W1).
**Declared risks:** the w=0.0 cells may dose constantly (frontier anchor, not a
policy of interest); PQN's wide-network stability argument favours the MLP
(arxiv-2407.04811 p.5–6); a gap averaged over the overlap can hide a crossing,
so the curves are always plotted.
**Artifacts:** `adaptive_therapy/policies/lstm.py` + test,
`research/w3_frontier.py`, `research/outputs/w3_*.json`, `w3_frontier.png`.
**Honest prior:** H_mem null (W1: a memoryless threshold explains ≥ 0.96 of the
MLP's doses); H_ct fails, given L424 and L461.

## [2026-09-24] — W3b: W3 re-registered on fresh seeds, with restricted-mean TTP as the frontier's y-axis and w = 0 removed

**Why.** W3 was INVALID ([L496](RESEARCH_LOG.md#L496)) and exposed a flaw in its
own rule: the MLP's capped median sat at the 357-d horizon in all 15 cells, so
the frontier could not rank it. **W3's data have been seen**, so this is a NEW
registration on **fresh seeds {3, 4, 5}**; no W3 shard enters a W3b verdict.
W3 is not re-graded.
**Hypotheses:** unchanged from W3 (H_mem: LSTM above MLP; H_ct: ODE-RNN above
LSTM, on Tier 1).
**Two changes, and only two.**
(1) **y-axis = restricted mean TTP (RMST)** at the 357-d horizon: the mean over
all 200 patients of min(TTP, horizon). **Declared deviation from PROJECT.md §10**
("median + IQR, not mean"): §10's reason is heavy tails, and a variable capped
at the horizon has none. The capped median, its IQR and censoring are still
reported for every cell, and the spec is not edited. Per-seed point = RMST of
the cell; the arm's point per w = median over seeds, as in W3.
(2) **w ∈ {0.05, 0.10, 0.20, 0.40}**: w = 0 removed. It rewards constant maximum
dosing and produced W3's invalid ODE-RNN cell (0.071, dose 355/357).
**Unchanged from W3:** arms (MLP 3090, LSTM-26 3045, ODE-RNN 3091 Q-path),
PQN Phase-2c settings, Tier 0 + Tier 1 cohorts (per-seed, as in W3), 200
patients, gap = mean y_A − y_B over the exposure overlap (100 points), 2000
paired patient-bootstrap resamples, closed-loop gate ≥ 0.10 per cell (Tier 1),
overlap ≥ 20% of both arms’ exposure ranges, pass iff the Tier-1 CI lies wholly
above 0, "robust" wording only if the Tier-1 CI lies above the Tier-0 gap.
**Grid:** 3 arms × 4 w × 3 seeds = 36 cells, ≈ 15 min on CPU.
**Declared risks.** (a) The overlap rule failed in W3 (16–19%) for reasons
unrelated to the endpoint: the LSTM's low-dose collapses widen its exposure
range. It may fail again; left unchanged here. (b) RMST is bounded by 357 d;
if both arms keep most patients progression-free, gaps will be small in days.
(c) Choosing the endpoint after seeing W3 is itself a selection event; fresh
seeds bound it, they do not remove it.
**Artifacts:** `research/w3_frontier.py w3b`, `research/outputs/w3b_*`.
**Honest prior:** H_mem and H_ct both fail; the W3 descriptives favour the MLP.

## [2026-09-24] — W3c: memory vs continuous time with per-cell validity and the trained objective as the primary statistic

**Why.** W3 and W3b were both INVALID ([L496](RESEARCH_LOG.md#L496),
[L526](RESEARCH_LOG.md#L526)) for two structural reasons, not one-off bad luck:
(a) a single open-loop cell voided a whole arm — recurrent arms fail the gate in
4/54 cells, so a per-arm rule invalidates almost every registration; (b) the
frontier overlap failed in 5 of 8 hypothesis × tier comparisons, because the
arms choose different doses and a frontier can only compare them where both
reach. **W3 and W3b data have been seen**: fresh seeds, and neither is re-graded.
**Hypotheses (unchanged):** H_mem — LSTM above MLP; H_ct — ODE-RNN above LSTM;
on Tier 1.
**Change 1 — per-cell validity.** A cell with greedy closed-loop < 0.10 (Tier 1)
is EXCLUDED and listed. An arm's point at a weight needs ≥ 3 of its 5 seeds
valid; an arm with < 3 valid seeds at any weight, or > 20% of its cells
excluded overall, is INVALID (a trainability result). Exclusion is decided once
on the full data, before any bootstrap.
**Change 2 — primary statistic = the trained objective.** Per patient,
U = min(TTP, 357) − w · cumulative exposure: the survival and exposure terms of
`envs/reward.py` (1 per day; w per unit dose-day), which every cell was trained
to maximise at its own w. Its burden-excess term is omitted: TTP already carries
it. Δ(A, B) = mean over the 4 weights of the mean paired difference U_A − U_B,
over patient × seed pairs where both cells are valid. It needs no exposure
overlap. **CI:** 2000 bootstrap resamples of the 200 patients (the same indices
for every cell). **H passes** iff the Tier-1 CI lies wholly above 0.
"Robust" wording only if the Tier-1 CI lies above the Tier-0 Δ point estimate.
**Change 3 — seeds {6, 7, 8, 9, 10}** (5, fresh), so one excluded cell leaves
a point.
**Change 4 — rule baselines, descriptive only:** none, MTD, AT50 and the 0.8 ×
baseline threshold (`policies/rules.py`), on the same cohorts and eval keys,
reported with U at each w. No test: they are reference points, not arms.
**Unchanged:** arms (MLP 3090, LSTM-26 3045, ODE-RNN 3091 Q-path), PQN
Phase-2c settings, Tier 0 + Tier 1 cohorts, 200 test patients, w ∈ {0.05, 0.10,
0.20, 0.40}. Grid 3 × 4 × 5 = 60 cells, ≈ 25 min on CPU.
**Secondary (descriptive):** W3b's RMST frontier and gaps where overlap ≥ 20%;
paired one-sided Wilcoxon on U per w; censoring; exclusions per arm.
**Kill.** H_ct fails ⇒ "no continuous-time edge over a matched recurrent arm on
Tiers 0/1", logged; W4 proceeds. An invalid arm ⇒ trainability result, no verdict
for the hypotheses that use it. No re-grade, no extra seeds after seeing results.
**Declared risks.** U adds days and w-weighted dose-days: it is the trained
objective, not a clinical utility, and its scale depends on w — hence the
mean over w is reported beside each w. Per-cell exclusion can flatter an arm
that fails often; the exclusion count is reported with every Δ.
**Artifacts:** `research/w3_frontier.py w3c`, `research/outputs/w3c_*`.
**Honest prior:** H_mem fails (W3b descriptive LSTM − MLP −45 d); H_ct is open
(W3b descriptive −5 d [−15, +12]).

## [2026-09-25] — W4: irregular observation — does continuous time help when visits are missed?

**Why.** Tier 4 is the one axis where continuous time has a mechanical reason to
help, left open by [CLOSED #1](../CLOSED.md) and W3c ([L565](RESEARCH_LOG.md#L565)).
No W4 data exist; W3c's policies have been seen on regular visits only.
**Missed visit (fixed here).** Each visit after the first is missed i.i.d. with
probability p. A missed visit gives no observation and no decision: the policy
state is carried, the next dt is the true gap, and the **last chosen dose is
given again** (hold: information is lost, not drug). Masks are nested across p
and identical for every arm and rule at a seed. TTP, exposure and burden
integral are scored on the weekly ground-truth grid. PQN takes its loss over
decisions only and bootstraps at the next decision (semi-MDP Q(λ)).
**Deviation from ROADMAP W4 ("train regular").** Trained on regular visits,
LSTM+Δt sees Δt = 7 throughout, so its Δt weight is a bias: the strawman the
ROADMAP warns against. The primary trains every arm at p_train = 0.35; the
ROADMAP design is secondary H_zs, on W3c's saved policies, at no training cost.
**Arms:** MLP-441, LSTM-26, LSTM-26+Δt (gap/7 d as a 2nd input; Q-path 3149,
+1.9%), ODE-RNN (3091; flow now in fixed 7/8-d steps so a 14-d gap is exactly
two 7-d flows, and a 7-d flow is unchanged). PQN Phase-2c settings. Grid 4 ×
w {0.05, 0.10, 0.20, 0.40} × **10 fresh seeds {11..20}** = 160 cells.
**Test:** the 200 Tier-0 test patients at p ∈ {0, 0.2, 0.5}, W3c's eval key.
**Validity (W3c rule):** a cell with greedy closed-loop < 0.10 at p = 0 is
EXCLUDED; an arm with < 60% valid seeds at any w or > 20% excluded is INVALID.
**H_ct4 (primary).** U = min(TTP, 357) − w·exposure. Δ(ODE, X) = mean over w
and p ∈ {0.2, 0.5} of the mean paired U_ODE − U_X. **CI: 2000 bootstrap
resamples of seeds AND patients** (W3c's patient-only CI is conditional on the
trained policies, [L596](RESEARCH_LOG.md#L596)). Passes iff the
95% CI lies above 0 for **every** X ∈ {MLP, LSTM, LSTM+Δt} (intersection–union).
**H_grace (secondary).** Same rule on the degradation U(p) − U(0), i.e. Q4's
"degrades more gracefully". Passing it without H_ct4 is worded "degrades less
but stays below", never "helps".
**H_zs (secondary).** W3c's 60 regular-trained policies (seeds 6–10) on the same
test, X ∈ {MLP, LSTM}, the H_ct4 rule. p = 0 must reproduce W3c's Tier 0
exactly (the script refuses otherwise).
**Descriptive:** Δ at p = 0 beside every Δ; patient-only CIs; per-seed Δ;
LSTM+Δt − LSTM; per-(w, p) one-sided paired Wilcoxon on U; per (arm, w, p)
capped median TTP + IQR, censoring, median exposure and burden integral; rules
none / MTD / AT50 / thr0.8 at each p.
**Kill.** H_ct4 fails ⇒ no continuous-time edge under missed visits either;
with CLOSED #1 and W3c, the negative holds on both axes. There is no LTC or
Neural-CDE rescue in this paper, and no extra seeds, rates or grid after seeing
results. An invalid arm ⇒ trainability result.
**Declared risks.** (1) Drops are i.i.d., so the next gap is unknown when
dosing, and Δt only describes the past, which a near-memoryless threshold
([L461](RESEARCH_LOG.md#L461)) may not need. (2) p_train sits between the
test rates, so p ≤ 0.5 is near-in-distribution; unseen gaps are H_zs's job.
(3) Relative to W3c, the ODE-RNN's training differs by float rounding (2e-7
relative in Q and gradients, from a traced gap). 300 updates amplify that into
a different policy. The MLP retrains bit-identically, and saved policies
evaluate identically. (4) At p = 0.35 there are ~65% as many decisions for the
same env-step budget.
**Compute:** ≈ 75 min on CPU, 6 cells in parallel (measured on the unregistered
seed 99, timing only: ODE-RNN 478 s per cell at p = 0.35 vs 204 s at p = 0, because
a vmapped batch pays its longest gap; MLP 40 s, LSTM+Δt 33 s).
**Artifacts:** `research/w4_irregular.py` →
`research/outputs/w4_irregular.{json,png}`, `w4_shards/`, `w4_policies/`.
**Honest prior:** H_ct4 fails. At regular visits the ODE-RNN sits ~100 d below
the MLP, and hold dosing costs every arm alike. H_zs is the likelier place for
an edge, and is still more likely to fail.

## [2026-09-29] — W4b: W4 re-registered intention-to-treat, with MTD as a reference arm every learned arm must beat

**Why.** W4 was INVALID ([L632](RESEARCH_LOG.md#L632)) because its closed-loop gate
excluded 25–52% of recurrent cells, and those excluded near-fixed schedules scored
*higher* U than the valid cells. At Tier 4 an open-loop schedule is a legitimate
policy, not a trainer collapse. The gate's original job (catch collapse, L312) is
now done by the seed+patient CI ([L596](RESEARCH_LOG.md#L596)): a collapsed seed
widens the CI; it no longer needs excluding. W4's data have been seen, so they
cannot be re-graded; W4b trains **fresh seeds {21..30}**.
**Unchanged from W4:** missed-visit semantics (hold), arms MLP-441 / LSTM-26 /
LSTM-26+Δt / ODE-RNN, PQN Phase-2c settings, p_train 0.35, w {0.05, 0.10, 0.20,
0.40}, 200 Tier-0 test patients at p ∈ {0, 0.2, 0.5}, U = min(TTP, 357) − w·exposure,
2000 seed+patient bootstrap resamples. 160 cells + 4 rules × 10 seeds.
**Validity (changed): intention-to-treat.** Every cell counts, whatever its
closed-loop score. The only exclusion is a non-finite rollout, and > 5% of an
arm's cells non-finite makes that arm INVALID. Closed-loop fraction is reported
per arm, descriptively.
**H_ct4b (primary).** Δ(ODE, X) = mean over w and p ∈ {0.2, 0.5} of the mean paired
U_ODE − U_X. Passes iff the 95% CI lies above 0 for **every** X ∈ {MLP, LSTM,
LSTM+Δt, MTD} (intersection–union). MTD is in the set because an edge over
learned arms that stays below a fixed schedule is not a scheduling edge.
**H_grace (secondary).** The same rule on U(p) − U(0), X ∈ {MLP, LSTM, LSTM+Δt}
(MTD's degradation is zero by construction).
**H_fb (secondary).** Does feedback still pay? The best learned arm by point Δ
vs MTD at p = 0.5 alone: CI above 0 ⇒ yes; below 0 ⇒ "at 50% missed visits a fixed
schedule beats every learned policy".
**Descriptive:** Δ at p = 0; per-seed Δ; LSTM+Δt − LSTM; per-(w, p) one-sided
paired Wilcoxon; per (arm, w, p) capped median TTP + IQR, censoring, median
exposure, median burden integral; closed-loop fraction; rules none / MTD / AT50 /
thr0.8. W4's cells under the ITT rule, labelled **post-hoc, not a test**.
**Kill.** H_ct4b fails ⇒ no continuous-time edge under missed visits; with LAW L1
the negative holds on both axes; the program closes and W6 is written. No extra
seeds, rates, arms (LTC, Neural CDE) or grid after seeing results.
**Declared risks.** W4's risks (1)–(4) carry over. (5) ITT lets a seed that
collapsed to a *bad* schedule pull its arm down; this is the intended reading:
unreliable training is part of the architecture's cost.
**Compute:** ≈ 75 min on CPU (W4 measured).
**Artifacts:** `research/w4b_itt.py` → `research/outputs/w4b_itt.{json,png,log}`,
`w4b_shards/`, `w4b_policies/`.
**Honest prior:** H_ct4b fails; the ODE-RNN was below the MLP in W4 (−42.8 [−76, −4]).
H_fb likely fails at p = 0.5 (W4: MTD 260 vs MLP 216).

## [2026-09-29] — S1: the decisive comparisons re-run on Strobl 2021's LV ranges (W3c's Tiers 0/1 and W4b's Tier 4)

**Why.** Only 9.8% of the provisional LV patients lie inside Strobl 2021's ranges
([L703](RESEARCH_LOG.md#L703)): our cost of resistance is ~70% against their 0–50%.
Every verdict so far holds only on that box, so none can reach a paper as
"Strobl-parameterised LV". This re-runs the two designs the paper rests on,
changing **only the θ box**.
**Cohort.** `sample_cohort(..., ranges="strobl2021")`: r_S 0.027/d and drug kill
1.5 fixed; r_R/r_S 0.5–1.0; one turnover d_T/r_S 0–0.5 on both populations;
f_R 0.001–0.1; n_0 0.1–0.75 (Table 1, PMC8455086 HTML). Tier-1 shift unchanged
(r_S, r_R × 1.5; K × 0.6). The provisional default reproduces bit-for-bit (hash-checked).
**Precondition, checked before registering (rules only, seed 0, no learned
policy):** Tier 1 untreated median TTP 14 d; MTD 224 d with 10% censored; thr0.8
312 d, 38%; no NaN. More headroom than the provisional box (MTD 357 d, 57%).
**Part A: W3c's design (regular visits, Tiers 0/1).** Arms MLP-441, LSTM-26,
ODE-RNN; PQN Phase-2c; w {0.05, 0.10, 0.20, 0.40}; **fresh seeds {31..40}**;
200 test patients per tier. **H_ct_S (primary):** Δ(ODE, X) = mean over w of the
mean paired U = min(TTP, 357) − w·exposure on **Tier 1**. Passes iff the 95%
seed+patient CI (2000 resamples) lies above 0 for **both** X ∈ {MLP, LSTM}.
Passing kills LAW L1 in this regime. Secondary: the same on Tier 0; LSTM − MLP (memory).
**Part B: W4b exactly** (train at p 0.35, Tier-0 test at p {0, 0.2, 0.5}, hold,
+LSTM+Δt), **fresh seeds {41..50}**. **H_ct4b_S** passes iff ODE's CI is above
0 for every X ∈ {MLP, LSTM, LSTM+Δt, MTD}; H_grace_S and H_fb_S as in W4b.
**Validity (both parts): intention-to-treat**, as in W4b. The only exclusion is a
non-finite rollout, and an arm with > 5% such cells is INVALID. This departs from
W3c's per-cell closed-loop gate (L632 showed the gate conflates collapse with
legitimate fixed schedules). Closed-loop fraction is reported descriptively.
**Descriptive:** per (arm, w, tier/p): capped median TTP + IQR, censoring, median
exposure, median burden integral; one-sided paired Wilcoxon per w; rules none /
MTD / AT50 / thr0.8.
**Kill / reading.** Neither ODE hypothesis passes ⇒ the negative replicates on
Strobl's ranges; the paper reports the Strobl regime as primary and the
provisional one as a second, high-cost-of-resistance regime. Either passes ⇒
the negative was an artefact of the provisional box; logged as such, with
no re-grade of earlier entries. No extra seeds, arms or grid after seeing results.
**Declared risks.** (1) PQN settings were tuned nowhere and may suit one regime
better. (2) r_S is fixed, so the Tier-1 shift moves r_S to 0.0405, further
outside training than under the provisional box. (3) f_R is sampled
log-uniformly (2 decades); Strobl sweeps it, and gives no sampling distribution.
**Compute:** ≈ 2 h on CPU (Part A 120 cells, Part B 160, W4b measured 87 min).
**Artifacts:** `research/s1_strobl.py` → `research/outputs/s1_strobl.{json,png,log}`,
`s1_shards/`, `s1_policies/`.
**Honest prior:** both fail. The MLP's threshold rule should survive, and a
weaker cost of resistance gives memory less to exploit, not more.

## [2026-09-29] — S2: full-state headroom bound — how much does seeing the hidden state gain over the burden-only MLP?

**Why.** LAW L1 explains the negative by assuming the current burden is near-
sufficient for dosing ([L461](RESEARCH_LOG.md#L461)). That assumption was never
measured. If seeing the hidden state gains little, memory had nothing to recover.
If it gains a lot, the recurrent arms (−44 to −50 d vs the MLP, [L766](RESEARCH_LOG.md#L766))
failed to learn what was there.
**Owner-approved exception to "burden-only" (2026-09-29).** The oracle is a
labelled reference, never an arm, and is never reported as a policy.
**Oracle.** S1's MLP-441 (LayerNorm, PQN Phase-2c) with a wider input: the same
noisy log-burden observation the arms get, plus noise-free log(S/b0), log(R/b0)
(**O_state**, 3 inputs); **O_full** adds θ as log(r_S/0.027), log(r_R/0.027),
log(K/b0), d_S/0.027, d_R/0.027, d_D (9 inputs). O_full's information contains
that of any burden-only policy, recurrent or not. Only the first layer grows.
**Pairing.** Same seeds, training cohorts, eval keys, observation noise and visit
masks as S1: Part A seeds {31..40}, Part B {41..50}, Strobl ranges. The
comparators are **S1's existing cells** (seen); only the oracle cells are new,
and no oracle setting is tuned.
**Primary (Part A, Tier 0, regular visits).** Δ = mean over w of the mean paired
U_Ofull − U_MLP, 95% CI over 2000 seed+patient resamples. Readings fixed now:
CI upper bound < **+10 d** ⇒ **no headroom** (the hidden state is worth < 10 d of
U to this trainer); CI lower bound > +10 d ⇒ **headroom** (the recurrent arms'
deficit is a learning failure); CI wholly < 0 ⇒ the trainer cannot use the
input, and no headroom claim either way; otherwise inconclusive. 10 d is a
quarter of the MLP's Tier-0 margin over MTD (40.4 d, S1).
**Secondary.** O_state vs MLP (Tier 0); both oracles at Tier 1 (descriptive:
the shift puts θ outside training); **Part B (35%-missed training, W4b design):**
Δ(O_full, MTD) at p = 0.5, with the same three readings at threshold 0: CI > 0 ⇒
full information restores feedback's value at 50% missed; CI < 0 ⇒ the collapse
is inherent to missed visits under hold; Δ(O_full, LSTM) at p 0.5 descriptive.
**Standing:** capped median TTP + IQR, median exposure, burden integral,
one-sided paired Wilcoxon (oracle > MLP) per w; ITT (non-finite only); 10 seeds.
**Kill.** No extra seeds, inputs, widths or trainers after seeing results; the
oracle is not re-run until it "works". A below-MLP oracle is reported as such.
**Declared risks.** (1) An empirical bound: PQN may not reach the full-information
optimum, so a small Δ bounds what *this trainer* extracts. (2) log(R/b0) reaches
−7 at f_R = 0.001; LayerNorm follows the first layer. (3) O_full's first layer
has 9× the MLP's input weights (Q-path ~6.6k vs 3.1k params).
**Compute:** 160 MLP-sized cells, ≈ 20 min on CPU.
**Artifacts:** `research/s2_headroom.py` → `research/outputs/s2_headroom.{json,log}`,
`s2_shards/`, `s2_policies/`.
**Honest prior:** inconclusive to small. The MLP ties a one-line rule, but only
the S/R split says when treatment has become futile, and burden reveals it late.

## [2026-09-29] — S3: LAW L1's headroom gate — does PQN at its published parallelism let the full-information reference beat the burden-only MLP?

**Why.** S2 ([L804](RESEARCH_LOG.md#L804)): the reference that also sees S, R
and θ scored 63 d below the burden-only MLP and earned less training reward even
on its own training patients, an optimisation failure. Our PQN collects 32
patients per update; the published setting is 1024 parallel environments
(arxiv-2407.04811 Table 4; noted in `pqn.py`). LAW L1 (amended) admits a new
trainer or budget only if, under it, this reference beats the MLP. S3 is that
gate and nothing else: no continuous-time arm runs here.
**The one change (trainer).** `PQNConfig(batch_patients=1024)`: 4 minibatches of
256 whole episodes, from a **2048-patient** training cohort per seed (was 200),
sampled without replacement. Unchanged from S2: 300 updates, lr, ε and λ
schedules, Strobl ranges, regular visits, MLP-441 LayerNorm, O_full's 9 inputs.
**Arms.** MLP and O_full, both retrained; w {0.05, 0.10, 0.20, 0.40}; seeds
{31..40}, the same test cohorts and eval keys as S1/S2.
**Primary.** Δ = mean over w of the mean paired U_Ofull − U_MLP at Tier 0, 95%
CI over 2000 seed+patient resamples. **R** = mean over the 40 cells of the
reference's minus the MLP's mean training return on its own 2048-patient
training cohort (S2's diagnostic, built into every cell). Readings, in order:
CI lower bound > **+10 d** ⇒ **headroom**: the gate opens, and a continuous-time
re-run under this trainer may be registered (not run here). Else R < 0 ⇒
**still trainer-limited**: the caveat stands as the paper's final word. Else CI
upper bound < 0 ⇒ **reward-limited**: the reference optimises the reward as
well but loses on U, so the reward (no TTP term, [L739](RESEARCH_LOG.md#L739))
binds. Else upper bound < +10 d ⇒ **no headroom**: L1's mechanism gains support.
Else inconclusive.
**Descriptive.** Tier 1; the budget effect on each arm vs its S1/S2 cells (same
seeds and test patients); capped median TTP + IQR, median exposure, burden
integral; one-sided paired Wilcoxon (O_full > MLP) per w; ITT, non-finite only.
**Kill.** No other budget (updates, lr, batch, cohort) after seeing results; a
trainer-limited result ends the budget search in this paper.
**Declared risks.** (1) Batch and training-cohort size change together; the gate
asks whether *some* budget works, not which part. (2) Gradient steps are
unchanged (300 × 4 × 4): if the failure is too few steps, 1024 patients per
step may not cure it. (3) R is a point estimate; its per-cell sign counts are reported.
**Compute:** measured on unregistered seed 99, timing only: 641 s (MLP) and 575 s
(O_full) per cell alone; 80 cells, 6 in parallel ≈ 2–3 h on CPU.
**Artifacts:** `research/s3_budget.py` → `research/outputs/s3_budget.{json,log}`,
`s3_shards/`, `s3_policies/`.
**Honest prior:** still trainer-limited or no headroom. The MLP gains from the
budget too, and 300 gradient steps may be the limit.

## [2026-09-30] — R1: reproduce Strobl 2021's published TTPs with our own LV vector field before building on its axes

**Why.** The monitoring study (M1, owner-approved 2026-09-30) will bin patients
on Strobl's cost and turnover axes, and the paper cites Strobl's ranges. We
verify rather than trust: our `lotka_volterra.vector_field` (no PK: u = d), fed
Table 1's parameters and the paper's two schedules, must land on its numbers.
**Setup (from the 1° review, pmc-8455086).** r_S 0.027/d, d_D 1.5, K = 1,
d_S = d_R = d̂_T·r_S, r_R = r̂_R·r_S; S0 = n0(1 − f_R), R0 = n0·f_R. CT: d = 1.
AT50: d = 1 until n < 0.5 n0, then 0 until n ≥ n0 (p.5). TTP = first day with
n > 1.2 n0 (`ttp_from_baseline`, confirm 0; p.6). RK4, 20 steps per day,
switching checked every step, 3000-d horizon.
**Targets.** Fig. 4E (p.24), Tumour 1 (n0 25%, f_R 0.1%): (i) d̂_T 0, r̂_R 1:
CT 303, AT 323 d; (ii) d̂_T 0, r̂_R 0.5: CT 575, AT 646; (iii) d̂_T 0.5, r̂_R 1:
CT 616, AT 848; (iv) d̂_T 0.5, r̂_R 0.5: no progression under either. Text p.6-7
(d̂_T 0, r̂_R 1), AT − CT at n0 25% / 75%: f_R 0.1%: 19 / 211 d; f_R 1%: 3 / 104 d;
f_R 10%: ≈ 0 at both.
**Pass iff** every finite target is within max(±3%, ±3 d); (iv) shows no
progression within 3000 d; and both f_R-10% gains are ≤ 3 d.
**Fail ⇒** log which targets miss and by how much. The paper's Fig. 5C framing
sentence is removed. M1 proceeds on our simulator, which never used Strobl's
numbers, and states the discrepancy. No re-tuning of our model to hit a target.
**Declared risk.** Their AT switches in continuous time; ours checks every
0.05 d, which can move a TTP by about a day.
**Artifacts:** `research/r1_strobl_repro.py` → `research/outputs/r1_strobl_repro.json`.

## [2026-09-30] — M1: when does monitoring stop paying? The missed-visit rate at which the best threshold policy falls to MTD, across cost of resistance and turnover

**Why.** At 50% missed visits no learned policy beat MTD in either regime
([L668](RESEARCH_LOG.md#L668), [L766](RESEARCH_LOG.md#L766)), and the margin
differed by regime. Strobl's cost and turnover axes decide whether adaptive
therapy pays, and R1 verified we reproduce its model ([L871](RESEARCH_LOG.md#L871)).
This is a question about the task: **no learned policy and no architecture**, so
it is outside CLOSED #2, whose bans on other drop rates and hold rules apply to
the architecture comparison only. Owner-approved 2026-09-30.
**Patients.** `strobl2021` ranges with r̂_R and d̂_T restricted to 10 bins:
r̂_R [0.25,0.5) [0.5,0.625) [0.625,0.75) [0.75,0.875) [0.875,1.0] × d̂_T [0,0.25)
[0.25,0.5]. The first r̂_R bin lies **outside Strobl's sweep**, the high-cost
region of its fitted patients, and is labelled so everywhere. n0 and f_R as in
S1. Per seed {1..10} and bin: a 200-patient selection cohort and a 200-patient
test cohort, disjoint draws.
**Policies (rules only).** MTD; threshold rules thr_x (full dose when burden >
x·baseline, else none), x ∈ {0.5, 0.6, …, 1.2}; AT50 descriptive. Weekly
visits, missed i.i.d. with p ∈ {0, 0.1, …, 0.6}. **Miss rule:** hold (the
protocol so far) or full (full dose at a missed visit). Per (seed, bin, p,
miss rule, w), the best x is chosen by mean U on the selection cohort and
scored on the test cohort. No selection happens on test patients.
**Statistic.** Δ_fb = U(best thr) − U(MTD), paired on test patients, mean over
w {0.05, 0.1, 0.2, 0.4}; 95% CI over 2000 seed+patient resamples.
**H_cost (primary).** At p = 0.5 under hold, Δ_fb differs between the
highest-cost Strobl bin (r̂_R [0.5, 0.625)) and the no-cost-end bin
([0.875, 1.0]). The CI of the difference must exclude 0 in **both** turnover
bins. Direction not predicted.
**H_miss (secondary).** At p = 0.5, U(best thr | full) − U(best thr | hold),
pooled over the 8 Strobl bins: CI > 0 ⇒ "the collapse is a hold-protocol
artefact"; CI < 0 ⇒ "dosing through a gap costs more than it saves".
**Descriptive.** A Δ_fb map (bin × p × miss rule); p*, the interpolated zero
crossing of Δ_fb per bin; chosen x; per bin at p 0.5: capped median TTP + IQR,
censoring, median exposure, burden integral; one-sided paired Wilcoxon (thr >
MTD); AT50.
**Kill.** H_cost fails ⇒ "within Strobl's range, cost of resistance does not
change what missed visits do to feedback". No bins, thresholds, rates or miss
rules added after seeing results.
**Declared risks.** (1) High turnover with high cost gives indefinite control
(Strobl p.9): U saturates at the horizon and Δ_fb → −w·ΔExposure; censoring is
reported. (2) The threshold family is the whole policy class; a better feedback
policy could exist. (3) Bins restrict r̂_R and d̂_T only.
**Compute:** ~27k rule evaluations of 200 patients, ≈ 30–60 min on CPU.
**Artifacts:** `research/m1_monitoring.py` → `research/outputs/m1_monitoring.{json,log}`, `m1_shards/`.

## [2026-09-30] — M1b: does M1's cost effect survive a three-year horizon?

**Why.** In M1's costliest bins MTD's median TTP sat at the 357-d cap with
75–100% of patients censored, so feedback could win only on exposure there, and
part of H_cost's effect may be the horizon ([L896](RESEARCH_LOG.md#L896),
caveat 1). Owner-approved 2026-09-30.
**The one change.** 156 weekly visits (horizon 1085 d) instead of 52. Identical
to M1 otherwise: bins, cohort seeds (the same patients), thresholds, p grid,
miss rules, selection/test split, w grid, statistic, bootstrap.
**H_cost_long (primary).** At p 0.5 under hold, highest-cost Strobl bin minus
no-cost-end bin. CI excluding 0 **with M1's (negative) sign** in both turnover
bins ⇒ "the cost effect is not a horizon artefact". Both CIs covering 0, or
either sign flipped ⇒ "M1's cost effect was at least partly the horizon".
Mixed ⇒ reported per turnover bin. Magnitudes in days are not comparable to
M1's, since U's range grows with the horizon; signs, CIs and p* are.
**Secondary.** H_miss as in M1; p* map; MTD censoring per bin at 1085 d.
Descriptive context: the share of test patients meeting Strobl's
indefinite-control criterion d̂_T > (1 − 1.2 n0)·r̂_R (p.9), which is derived
for Strobl's baseline progression rule, not our nadir rule. Measured on seed-1
test cohorts: 4–25% at low turnover, 38–95% at high.
**Kill.** No further horizons. The result enters the paper either way,
replacing caveat 1 with a measured statement.
**Compute:** ≈ 3× M1 (M1 ran about 25 min) ≈ 75–90 min on CPU.
**Artifacts:** `research/m1b_horizon.py` → `research/outputs/m1b_horizon.{json,log}`, `m1b_shards/`.

## [2026-09-30] — M1c: do M1/M1b's monitoring findings hold for the median patient?

**Why.** PROJECT.md §10 asks for medians, never means, and M1b showed they can
disagree: in the no-cost high-turnover bin the mean-U winner has a *shorter*
median TTP than MTD (196 vs 266 d, [L932](RESEARCH_LOG.md#L932)). **Post-hoc**
on M1/M1b's saved per-patient shards; no new simulation. Declared selection
risk: that one bin's medians were seen before this registration.
**Statistic.** Δ_med = median U(best thr) − median U(MTD) over pooled test
patients (10 seeds × 200), mean over w {0.05, 0.1, 0.2, 0.4}. Best x chosen per
(seed, bin, p, miss rule, w) by **median** U on the selection cohort. CI: 2000
seed+patient resamples, both arms resampled jointly, as in M1.
**P1 (primary).** M1b (1085 d), p 0.5, hold, the 4 high-turnover Strobl bins:
Δ_med CI > 0 in all 4 ⇒ "the three-year reversal holds for the median
patient"; CI < 0 or covering 0 in all 4 ⇒ "the reversal is a mean/tail
effect"; otherwise mixed, reported per bin.
**P2.** H_cost under Δ_med at both horizons (M1's bins, p 0.5, hold): CI < 0 in
both turnover bins ⇒ "the cost effect holds for the median patient".
**Descriptive.** Δ_med map and p*_med per bin, both horizons, both miss rules;
Δ_med with M1's mean-selected x; capped median TTP + IQR and median exposure.
**Kill.** No other statistic (trimmed means, quantiles other than the median)
added after seeing results. Enters the paper either way.
**Artifacts:** `research/m1c_medians.py` → `research/outputs/m1c_medians.{json,log}`.

## [2026-09-30] — W5-pre: does treated Hahnfeldt / Kuznetsov have TTP dynamic range?

**Why.** ROADMAP W5 and Stage 2 (M1's monitoring map across mechanisms) both
need it; untreated, both families progress within 1–2 visits with zero IQR
([L88](RESEARCH_LOG.md#L88)). Rules only, no learning. A diagnostic gate, not an
architecture or monitoring result.
**Cohort.** Provisional θ ranges (the only ones these families have; sources
2°), 200 patients × seeds {1, 2, 3} per family, cohort seed 300_000 + seed.
**Policies.** none, MTD, AT50, thr_x for x ∈ {0.5 … 1.2} (M1's set). Weekly
visits, horizons 357 d (primary) and 1085 d; p ∈ {0, 0.5}, miss rule hold.
**Pass, per family, at 357 d, p 0 — all three:** (P1) MTD capped TTP median
≥ 28 d (not resolution-limited); (P2) spread across patients under MTD or the
best thr: IQR width ≥ 14 d and censoring in [5%, 95%]; (P3) policies separate:
some thr_x vs MTD, paired over the 600 patients, has |median paired ΔTTP| ≥ 7 d
with two-sided Wilcoxon p < 0.01. Every trajectory finite, or the family fails.
**Reported.** Per family × policy × horizon × p: capped median TTP + IQR,
censoring, median exposure, median burden integral.
**Kill.** A family that fails is cut from W5 and from Stage 2, stated as a
limitation. No change to θ ranges, PK/effect, TTP rule, visit interval or
policy set to rescue it; re-opening needs 1°-read literature ranges (new data).
**Artifacts:** `research/w5_precondition.py` → `research/outputs/w5_precondition.{json,log}`.

## [2026-09-30] — T1: with 10× the gradient steps, does the full-information reference beat the burden-only MLP?

**Why.** S3 ([L840](RESEARCH_LOG.md#L840)) closed the budget search *in the
paper* and named "more gradient steps" as a new registration; its declared risk
(2) was that steps were unchanged. Owner-approved 2026-09-30 as a follow-up
program; reported as such, never as a re-run of S3.
**The one change.** `n_updates` 300 → **3000**; ε decay and lr decay keep their
S3 *fractions* of training (ε over the first 10%, lr linear to 0). Everything
else is S3: 1024 patients per update from a 2048-patient cohort, Strobl
ranges, regular visits, MLP-441 LayerNorm, O_full's 9 inputs, w {0.05, 0.10,
0.20, 0.40}, seeds {31..40}, S1/S2/S3's test cohorts and eval keys.
**Primary and readings:** S3's, unchanged and in the same order (Δ U O_full −
MLP at Tier 0, seed+patient CI; R = training-return difference): lo > +10 d ⇒
headroom, the gate opens; else R < 0 ⇒ still trainer-limited; else hi < 0 ⇒
reward-limited; else hi < +10 d ⇒ no headroom; else inconclusive.
**Descriptive.** Tier 1; step effect per arm vs its S3 cells (same seeds and
patients); capped median TTP + IQR, median exposure, burden integral;
one-sided Wilcoxon per w; MLP vs thr0.8.
**Kill.** Still trainer-limited ⇒ PQN is retired as this program's trainer; the
next attempt is another algorithm, never more steps, batch or lr. Headroom ⇒ a
continuous-time re-run under this trainer may be registered (not run here).
**Compute:** not timed; ≈ 10× S3's 690 s per cell, 80 cells, 6 in parallel ≈
25–30 h on CPU.
**Artifacts:** `research/t1_steps.py` → `research/outputs/t1_steps.{json,log}`, `t1_shards/`, `t1_policies/`.
**Honest prior:** S3 closed a third of the gap with R near 0, so a narrower gap
is likely; the +10 d margin is not.

## [2026-09-30] — R2: reproduce Strobl 2021's LV fits to the Bruchovsky IADT cohort before replaying policies on them

**Why.** Owner-approved real-data step (PROJECT.md §11 "calibration touch"). R1
([L871](RESEARCH_LOG.md#L871)) verified Strobl's simulations, not its patient
fits (Fig. 5C). Fitted patients are only usable as replay patients if the fit
itself reproduces. Protocol from Strobl p.6 (1°).
**Data.** `data/bruchovsky/dataTanaka/Bruchovsky_et_al` (Tanaka; gitignored),
72 patients minus the 5 metastatic ones in `classificationsofpatients.txt` = 67,
Strobl's count. Relapse labels from that file.
**Fit (Strobl p.6).** Our `lotka_volterra.vector_field`, K = 1, r_S = 0.027,
d_D = 1.5, u = the file's on/off flag held until the next row. Residual:
PSA(t)/PSA(0) − n(t)/n0 over rows with a PSA value. Free, bounded: r̂_R [0,1],
d̂_T [0,1], n0 [0.1,1], f_R [1e-5, 0.25]. 25 seeded restarts (seed 0);
best = min RSS (AIC with k fixed). **Declared deviations:** bounded
trust-region least squares (scipy `trf`) instead of lmfit's LM; RK4 at 10
steps/day; PSA(0) = first recorded PSA.
**Pass — all three, else "fails to reproduce" per criterion:** (F1) patients
with r² < 0.5 excluded: 10–20 (Strobl 15); (F2) Pearson(r̂_R, d̂_T) over kept
patients ≥ 0.5 (Strobl 0.77); (F3) r̂_R + d̂_T separates relapse from
non-relapse, two-sided Mann-Whitney p < 0.01 (Strobl U = 82).
**Descriptive.** Fitted ranges; share inside Strobl's sweep box and in each M1
bin; share meeting indefinite control; restarts landing within 1% of the best
RSS with r̂_R differing by > 0.1 (non-identifiability, Phase 0 [L48](RESEARCH_LOG.md#L48)).
**Kill.** > 25 patients excluded ⇒ no replay on these fits. No change to
bounds, residual, restarts or the exclusion rule after seeing fits.
**Artifacts:** `research/r2_bruchovsky_fit.py` → `research/outputs/r2_bruchovsky_fit.{json,log}`.
**AMENDMENT 2026-09-30 (before any fit ran):** rows are sorted by day; patients 016 and 037 each have one empty-PSA row out of date order in the source file.

## [2026-10-01] — P1: do the paper's two negatives hold on the 55 Bruchovsky-fitted patients?

**Why.** Owner-approved real-data replay (PROJECT.md §11). R2
([L1075](RESEARCH_LOG.md#L1075)) gives 55 fitted parameter vectors. Each is a
model fitted to one patient's PSA, not the patient; no clinical claim.
**Cohort.** The 55 R2 fits with r² ≥ 0.5, fixed: θ = (0.027, r̂_R·0.027, 1e4,
d̂_T·0.027, d̂_T·0.027, 1.5), y0 from (f_R, n0). The same 55 for every seed.
**Policies, no retraining.** S1's saved policies: Part A (MLP, LSTM, ODE-RNN,
seeds 31–40, regular-visit trained) and Part B (+ LSTM+Δt, seeds 41–50, trained
at 35% missed); rules MTD, AT50, thr0.8, none. Episode, PK, U and the
seed+patient CI as S1; ITT (non-finite cells only excluded).
**H1 (Part A, p 0):** ODE−MLP and ODE−LSTM. Neither CI lower bound > 0 ⇒ "no
continuous-time edge replicates on fitted patients"; both > 0 ⇒ "an edge
appears on fitted patients"; else mixed.
**H2 (Part B, p 0.5):** each learned arm − MTD. No CI lower bound > 0 ⇒ "no
learned arm beats MTD at 50% missed, replicated"; else name the arm.
**Descriptive.** MLP − thr0.8 at p 0; per arm capped median TTP + IQR, censoring,
median exposure, burden integral; one-sided Wilcoxon; H1/H2 restricted to the
20 fits inside Strobl's sweep box.
**Declared risks.** (1) Fits use linear drug and daily dosing; replay uses our
PK and weekly visits. (2) 64% of fits lie outside the training box, so this
is also a shift. (3) θ is weakly identified from PSA ([L48](RESEARCH_LOG.md#L48)).
**Kill.** Enters the paper either way. No change to the cohort, exclusion rule,
policy set or statistic after seeing results.
**Artifacts:** `research/p1_replay.py` → `research/outputs/p1_replay.{json,log}`.

## [2026-10-02] — V1: do the paper's architecture negatives survive three reviewer objections, using saved policies only?

**Why.** ROADMAP V-track. Three anticipated objections to `paper/main.tex`,
each answerable from S1's saved policies without retraining: (a) the reward has
no TTP term ([L739](RESEARCH_LOG.md#L739), [L1042](RESEARCH_LOG.md#L1042)), so
the memory arms may optimise it *better* and still lose U; (b) U is a mean,
and a heavy tail could carry it; (c) "no learned arm beats MTD at 50% missed" is
a one-year statement ([L932](RESEARCH_LOG.md#L932)). **Post hoc** on S1's
policies and test cohorts, registered before any V1 number is computed.
**V1a — reward vs U per arm.** S1 Part A policies (MLP, LSTM, ODE-RNN, seeds
31–40, 4 w). Per cell, mean episode return under the cell's own reward on the
seed's 200-patient training cohort, and U on S1's Tier-0 test cohort; eval key
and `score` as S2's diagnostic. Statistic per contrast X − MLP (X ∈ {LSTM,
ODE-RNN}) and ODE − LSTM: ΔR, the mean training-return difference over the 40
cells, with a CI resampling seeds (2000 draws); cells with ΔR > 0 counted.
Readings, per contrast whose ΔU CI (S1) excludes 0: ΔR CI upper < 0 ⇒ "loses
the reward too: not reward misalignment"; ΔR CI lower > 0 ⇒ "misaligned: wins
the reward, loses U"; else "ties the reward". Descriptive: Part B's four arms on
their training cohorts at drop 0.35.
**V1b — rank robustness.** S1 shards only, no rollouts. For every contrast the
paper reports (Part A Tier 0/1: ODE−MLP, ODE−LSTM, LSTM−MLP; Part B p 0.5:
each learned arm − MTD, ODE−MLP, ODE−LSTM): the median over (seed, patient) of
the paired U difference per (w, p), averaged over w, with the same seed+patient
bootstrap as `w4.delta_draws`. Per contrast: "agrees" if mean and median CIs
exclude 0 on the same side or both cover 0; "weaker" if one excludes 0 and the
other covers it; "reversed" if they exclude 0 on opposite sides. **Pass:** no
contrast reversed, and the paper's headline readings (ODE not above MLP or LSTM
at Tiers 0/1; no learned arm's lower bound > 0 against MTD at p 0.5) unchanged.
Descriptive: per arm, share of test patients censored at 357 d.
**V1c — three-year replay.** S1 Part B policies (trained on 52-visit episodes at
35% missed) and MTD, thr0.8, rolled out unchanged for 156 visits (1085 d, M1b's
horizon) on S1's Tier-0 test cohorts and eval keys, drops {0, 0.2, 0.5}; U with
the 1085-d cap. Primary: each learned arm − MTD at p 0.5, seed+patient CI.
Readings: any lower bound > 0 ⇒ "a learned arm beats MTD over three years: the
one-year statement is horizon-bound for learned policies"; every upper bound < 0
⇒ "learned arms stay below MTD over three years"; else "no arm above MTD".
Descriptive: ODE−MLP and ODE−LSTM at p 0.5; the V1b median version; capped
median TTP + IQR, censoring, exposure, burden integral per arm.
**Declared risks.** (1) V1c runs every policy 3× past its training horizon; a
recurrent state may drift, so a negative is weak evidence and a positive is the
informative reading. (2) V1a's training return is on the training cohort the
policy saw; it measures optimisation, not generalisation. (3) Medians of paired
U differences are dominated by exposure where both arms are censored.
**Kill.** Enters the paper (or its appendix) whatever the reading. No change to
cohorts, keys, policies or statistics after seeing a number. A reversed V1b
contrast is reported in the main text, not the appendix.
**Artifacts:** `research/v1_review.py` → `research/outputs/v1_review.{json,log}`, `v1_shards/`.
**Compute:** V1a 120 + 160 policy evaluations, V1c 160 × 3 drops at 3× length;
one process per (part, arch) (L207); ≈ 1 h on CPU.

## [2026-10-02] — V2: with a survival term that stops at progression, does the full-information reference beat the burden-only MLP?

**Why.** T1 ([L1042](RESEARCH_LOG.md#L1042)) read "reward-limited": the
reference ties the MLP on training reward and still trails on U by 20 d. No term
of the reward is TTP ([L739](RESEARCH_LOG.md#L739)), though the spec's reward is
"+TTP" ([PROJECT.md §7](../PROJECT.md)). ROADMAP V2. Owner-approved 2026-10-02.
**The one change.** `RewardConfig(survival_until="progression")`: the survival
term (1/day) is paid until progression is **confirmed online** —
`envs/reward.progression_update`, the running-nadir rule with its 28-d window.
The survival total is then TTP + 28 d, except for progressions in the last 28 d
of the horizon (tested: `test_online_progression_confirms_exactly_one_window_after_the_offline_ttp`).
Chosen over the unconfirmed 1.2 × nadir crossing because that crossing can
recover and is not the endpoint; chosen over a retroactive TTP reward because
the reward at a visit must not depend on later visits. Burden and exposure
terms unchanged. Everything else is T1: PQN 1024 patients × 3000 updates,
2048-patient cohort, Strobl ranges, regular visits, MLP-441 LayerNorm and O_full
(9 inputs), w {0.05, 0.10, 0.20, 0.40}, seeds {31..40}, S1's test cohorts and
eval keys.
**Primary and readings:** T1's, unchanged and in the same order (Δ U O_full −
MLP at Tier 0, seed+patient CI; R = training-return difference under V2's
reward): lo > +10 d ⇒ headroom, the gate opens; else R < 0 ⇒ trainer-limited;
else hi < 0 ⇒ still reward-limited; else hi < +10 d ⇒ no headroom; else
inconclusive.
**Descriptive.** Tier 1; reward effect per arm vs its T1 cell (same seeds,
patients); capped median TTP + IQR, median exposure, burden integral; one-sided
Wilcoxon per w; MLP vs thr0.8.
**Kill.** Headroom ⇒ V3 (S1 Part A's architecture comparison under V2's reward)
may be registered. No headroom ⇒ LAW L1's mechanism ("the current observation
is enough") gets its first direct support; no continuous-time re-run.
Trainer-limited or still reward-limited ⇒ the reward search ends in this paper:
no third reward, no reweighting of the burden or exposure terms.
**Declared risks.** (1) The survival term now rewards dosing that delays
progression, so both arms may dose more; exposure is reported beside U.
(2) The burden term still pays for burden below the progression line, so V2 is
"T1 + a TTP term", not a pure TTP reward. (3) The policies do not observe the
run counter; for a burden-only arm the reward depends on unobserved state, as
the burden term already does through the nadir.
**Compute:** T1's shards: median 6677 s per cell, 80 cells, 148 CPU-h; 6 in
parallel ≈ 25–30 h.
**Artifacts:** `research/v2_ttp_reward.py` → `research/outputs/v2_ttp_reward.{json,log}`, `v2_shards/`, `v2_policies/`.
**Honest prior:** the reference's T1 deficit came with less drug and more
burden, which the TTP term now charges for, so the gap should narrow. Whether
it clears +10 d is open.

## [2026-10-02] — V5: do our simulators reproduce Hahnfeldt 1999 and Zhang 2017 from their own pages?

**Why.** ROADMAP V5. W5-pre ([L1004](RESEARCH_LOG.md#L1004)) failed on a
Hahnfeldt whose drug scale and initial state were ours; a reviewer reads it as a
brittle task. Zhang 2017 is the candidate second family with a resistance
mechanism. Both PDFs now read `1°` (2026-10-02). As R1 did for Strobl
([L871](RESEARCH_LOG.md#L871)): reproduce published numbers before any policy
touches the model. Rules and fixed schedules only; no learning.
**V5a — Hahnfeldt 1999** (Cancer Res 59:4770). `envs/hahnfeldt.vector_field`
unchanged, driven with u = 1 and e replaced by e·g(t). g is the paper's C3 with
boli: g jumps by D at each injection and decays at clr (p.4773). Table 1
(p.4771): λ1 0.192, λ2 0, b 5.85, d 0.00873, K0 625; (e, clr) TNP-470 (1.3, 10.1),
endostatin (0.66, 1.7), angiostatin (0.15, 0.38). Targets:
(H1) untreated set point ≈ 17,400 mm³ (p.4774); (H2) TNP-470 30 mg/kg q.o.d. set
point ≈ 12,300 mm³, the same from V0 = 170, 12,300 and 17,300 (p.4774, Fig. 4 II
B–D); (H3) angiostatin 14 mg/kg/day from 177 mm³: overshoot, then set point
≈ 240 mm³ (p.4774); (H4) TNP-470 from V0 = 200, K0 = 625: day-13 volume 1840 mm³
under q.o.d. boli vs 1300 mm³ at the same continuous dose (p.4773); (H5)
endostatin 20 mg/kg/day regresses, while endostatin 4 mg/kg/day and TNP-470
30 mg/kg q.o.d. do not control (p.4773), read at day 15 against V0 = 200.
Set point = mean V over the last 100 d of a 2000-d run. Tolerance: 3% for H1–H3
and 10% for H4 (its V0 is "≈200", p.4772).
**V5c — Zhang 2017** (Nat Commun 8:1816). dx_i/dt = r_i x_i (1 − Σ_j α_ij x_j / K_i)
(Eq. 1, p.7); r = 0.1 × (0.278, 0.355, 0.665) /day (p.7); K_T+ = 1.5 x_TP off drug
and 0.5 x_TP on; K_TP = 10,000 off and 100 on; K_T− = 10,000 (p.7). α from Table 3
(p.8), row i, column j, consistent with all six inequalities on p.7.
dPSA/dt = Σ x_i − 0.5 PSA (Eq. 2, p.8). Initial state: equilibrium frequencies,
total cells and PSA at 25% of the untreated equilibrium. Run untreated until PSA
reaches 80% of equilibrium, then treat (p.8). Adaptive: stop below 50% of that
PSA, resume at or above it (p.8). Targets (Table 1, p.7), **Representative
Patient #2**: 5% T− at 323 under every regimen; 90% T− at 784 under MTD; under
adaptive, 90% T− never reached ("indefinite"; checked to 10,000).
Tolerance: R1's max(3%, 3 d).
**Declared gaps, not failures.** Patient #1 starts with no T− at equilibrium,
and the main text gives no rule for how T− arises (p.7, Table 1 note). If its
equilibrium T− is exactly 0, its rows are reported as not reproducible from the
main text. The metronomic schedule's durations are not given (p.3, p.8), so it
is skipped. The Supplementary Information was not obtained.
**Pass/kill.** V5a: all of H1–H5 ⇒ reproduces; a W5-pre re-run in Hahnfeldt's
units and daily dosing may then be registered (a mouse timescale of days, so
not on our weekly-visit episode). Any fails ⇒ Hahnfeldt stays closed, and the
paper says why. V5c: Patient #2's three targets ⇒ reproduces; a replay of S1's
policies on Zhang patients may be registered. Otherwise no second-family
claim. No change to parameters, initial-state rule, tolerances or progression
readings after a run.
**Artifacts:** `research/v5_repro.py` → `research/outputs/v5_repro.{json,log}`.

## [2026-10-02] — V5d: with Zhang 2017's Supplementary Information, does one declared reading of the unstated details reproduce Supplementary Table 3?

**Why.** V5c failed from the main text ([L1176](RESEARCH_LOG.md#L1176)); its
re-open condition was the SI (new data), now on disk as
`literature/pdf/url-zhang-2017-mcrpc-at-SI.pdf` (3 pp.). The SI adds targets
(Supp. Tables 1 and 3, all 22 competition orderings; Patient #1 = row 7,
Patient #2 = row 5, p.3), but **not** the time unit, the integration scheme or
the seeding of T−. So V5d declares a small reading space up front and asks
whether exactly one reading fits many independent numbers.
**Part 1 — structure, no time axis.** (S1) Assign {0.4, …, 0.9} to the six
off-diagonal α_ij (row i, column j) under the six inequalities of p.7; exactly
22 assignments ⇒ our reading of the inequalities is the paper's. (S2) The
untreated equilibrium frequencies of the 22 (V5's integration, k = [1.5 TP,
10,000, 10,000]) match Supp. Table 1's second column as a multiset, by optimal
assignment, within ±2 percentage points per cell (the table is rounded to
integers). Table 3's two matrices must land on rows 7 and 5. Fails ⇒ the model
as transcribed is not the paper's, and Part 2 is not run.
**Part 2 — time readings, declared now.** R-a: ODE, r = 0.1 × cell-line rates
(V5's reading). R-b: ODE, r = 0.01 ×. R-c: discrete-time map, one step per unit,
x ← x + r x (1 − Σα x/K) and PSA ← PSA + Σx − 0.5 PSA ("50% of the PSA decays
out of the serum each time step", p.8), r = 0.1 ×. R-d: R-c with r = 0.01 ×.
**R-b and R-d were suggested by V5's 7–10× gap; that is declared selection.**
Initial state and treatment start as V5 (p.8); adaptive: off below 50% of the
start PSA, on at or above it; horizon 10,000 units. Targets: every
row of Supp. Table 3 whose Supp. Table 1 equilibrium has T− > 0 (rows 12, 5, 2,
6, 18, 1, 4, 3, 11, 17), MTD and adaptive, 5% and 90% T−: 40 cells, where "---"
means not reached by 10,000. Rows are matched to our orderings by Part 1's
assignment. Rows with T− = 0 are excluded: the SI gives no seeding rule.
A cell is hit if within max(3%, 3 units), or "---" on both sides.
**Pass.** A reading passes at ≥ 36 of 40 cells (90%). Exactly one passes ⇒ it
becomes the Zhang simulator, and a replay of S1's policies may be registered.
None ⇒ Zhang 2017 is closed as not reproducible from paper + SI. More than one
⇒ ambiguous: no simulator is adopted.
**Kill / no-move.** No reading outside R-a–R-d, no change to tolerance, rows or
horizon after a run. Metronomic is skipped (schedule durations not given).
**Artifacts:** `research/v5d_zhang_si.py` → `research/outputs/v5d_zhang_si.{json,log}`.

## [2026-10-02] — V5e: V5d's Part 2, gated on the structure checks that test our transcription

**Why.** V5d ([L1216](RESEARCH_LOG.md#L1216)) failed its Table-3 check because
the paper contradicts itself: Patient #1's printed matrix is SI row 22, not
row 7. S1 (22 orderings) and the 22-row equilibrium match passed, so the
transcription is the paper's. **This gate change was decided after seeing V5d's
Part 1**; owner-approved 2026-10-02. Patient #1 (row 7, T− = 0) is not among
Part 2's rows, so the failing check touches none of its targets.
**Design.** V5d's Part 2 exactly: readings R-a–R-d, the 40 cells of Supp.
Table 3 (rows 12, 5, 2, 6, 18, 1, 4, 3, 11, 17 × MTD/adaptive × 5%/90%), rows
mapped to orderings by V5d's equilibrium assignment, hit within max(3%, 3
units) or "---" on both sides, pass at ≥ 36/40. Exactly one passing reading ⇒
the Zhang simulator; none ⇒ closed as not reproducible from paper + SI; more
than one ⇒ ambiguous, none adopted.
**No-move.** As V5d. No fifth reading, whatever the scores.
**Artifacts:** `research/v5d_zhang_si.py --v5e` → `research/outputs/v5e_zhang_times.{json,log}`.

## [2026-10-02] — O1: how much U is there to gain with full information, measured without any trainer?

**Why.** LAW L1's mechanism ("the current observation is enough") is untested
because every full-information reference so far was trained by PQN, which
could not use extra state (S2 [L804](RESEARCH_LOG.md#L804), S3, T1
[L1042](RESEARCH_LOG.md#L1042)). At regular visits the simulator is
deterministic given the doses. The full-information optimum is therefore the
best 52-visit dose sequence for each patient, and any sequence a search finds
bounds that optimum from below. Owner-approved 2026-10-02 (workshop-paper
priority 1).
**Oracle.** Per test patient and per w, search dose sequences in {0, ½, 1}^52
on the true simulator (true θ and state, no observation noise; PK, visits and
horizon as S1). (i) *Tuned threshold*: the best of ThresholdPolicy levels
0.30–1.50 by 0.05, plus none and MTD, chosen on that patient. (ii) *Sequence
search*: cross-entropy over per-visit dose distributions, warm-started at
(i)'s sequence (probability 0.8 on its dose), population 128, 25 iterations,
elite 10%, smoothing 0.7, seed 0. Search objective: V2's online survival
(RewardConfig(survival_until="progression"), burden term 0) − w·dose. The
returned sequence is the best evaluated, and is **scored with the official U**
(`ttp_running_nadir`, `cumulative_exposure`).
**Cohort.** S1 Part A's Tier-0 test cohorts, seeds 31–40 (2000 patients), w
{0.05, 0.10, 0.20, 0.40}.
**Primary.** Δ_O = U(oracle) − U(MLP), against S1 Part A's saved MLP shards
(same patients), mean over w, seed+patient CI (`w4.delta_ci`). lo > +10 d ⇒
**headroom**: full information is worth more than 10 d, and the MLP's limit is
learning, not the task. hi < +10 d ⇒ **no headroom found**: a full-information
search gains less than 10 d over the burden-only MLP, LAW L1's mechanism gets
its first direct support (a lower bound: the search is not proven optimal).
Else inconclusive. +10 d is L1's margin.
**Secondary.** Δ_T = U(tuned threshold) − U(MLP): how much of any gain is
patient-specific threshold tuning alone.
**Descriptive.** Per-w Δ_O; share of patients gaining > 10 d; oracle median
exposure, capped median TTP + IQR, burden integral; search gain over (i); mean
improvement over the last 5 iterations (a large value means the bound is loose).
**Declared risks.** (1) Lower bound only: "no headroom found" is weaker than "no
headroom". (2) The search objective is TTP + 28 d − w·exposure, not U exactly;
the scoring is exact. (3) Tier 0 only; Tier 1 and missed visits are not covered.
**Kill.** Enters the paper either way. No change to search settings, grid,
cohort or margin after a run.
**Compute.** Measured: 0.67 s per 1000 rollouts under V2's load; ≈ 26M rollouts
≈ 5 h CPU.
**Artifacts:** `research/o1_oracle.py` → `research/outputs/o1_oracle.{json,log}`, `o1_shards/`.

## [2026-10-02] — O1b: is O1's headroom full information, or one fixed threshold the MLP did not learn?

**Why.** O1 read "headroom" (oracle − MLP +23.1 d [+20.3, +25.8]); its tuned
thresholds cluster at 1.15 × baseline (IQR 1.05–1.20), far from the MLP's
learned ~0.8 (seen in O1's descriptive output, **so this split is registered
after seeing O1**). If one fixed, burden-only level captures most of the
gain, the headroom is not about information. It is a memoryless rule that PQN's
reward did not lead to.
**Design.** O1's cohorts, levels (0.30–1.50 by 0.05) and noise-free rule
rollouts (re-run; deterministic). *Global rule*: per w, the level maximising
mean U on the other nine seeds' cohorts, scored on the held-out seed (leave
one seed out); burden-only, memoryless, no information about the test patient.
Statistics, seed+patient CI (`w4.delta_ci`): Δ_G = U(global) − U(MLP), the
**learnable-without-memory** part; Δ_I = U(tuned) − U(global), the **value of
patient-specific information**. The global rule here is noise-free; its score
with S1's 5% observation noise is reported beside it (descriptive).
**Readings.** Δ_I hi < +10 d and Δ_G lo > +10 d ⇒ "the headroom is a fixed
threshold PQN's reward did not reach, not information". Δ_I lo > +10 d ⇒
"patient-specific information is worth > 10 d: memory has something to
recover". Else mixed, report both.
**No-move.** No other policy class, level grid or selection rule after a run.
**Artifacts:** `research/o1b_split.py` → `research/outputs/o1b_split.{json,log}`.

## [2026-10-04] — V6-gate: does our Baez–Kuang/Phan 2-population Droop model fit our Bruchovsky patients as well as Phan 2019 reports?

**Why.** ROADMAP V6 step 2. Phan et al. 2019 (`doi-10.3934-mbe.2019010`, read `1°`
2026-10-04) fit the 2- and 3-population cell-quota models to the Vancouver IAS
data. Their recommended model is the 2-population model (A1, p.200): eqs.
2.1–2.6 (p.189), with AD → AI by irreversible transformation λ(Q). We hold 72 of the
109 IDs (`data/bruchovsky/dataTanaka/`), including all four of their example patients.
**Model and fit (Phan p.193).** x1, x2, Q, P. γ1 = 20 is fixed. u comes from the on/off column (u = 0 on).
Free parameters within Table 1 (p.192), restricted to the 2-population parameters: μ, q1, q2, b, σ, ε,
d1, d2, δ1, δ2, R1, R2, c, K, γ2 and Qm, plus x1(0), x2(0) in [1e-2, 1e4] mm³ (Fig. 4, p.200, shows up to ~4000;
log-uniform). Q(0) and P(0) are the first observed values. The objective is
MSE_PSA + MSE_androgen over the first 1.5 cycles (on, off, on). **Declared deviation:**
Phan chose initial guesses by hand; we use 30 seeded log-uniform restarts and
`scipy.optimize.least_squares` with the bounds. The best restart is kept.
**Cohort.** The stated rule (≥ 20 PSA and ≥ 20 androgen points in 1.5 cycles,
p.189) admits 43/72 patients, not 62. Of these, 27 have the further full cycle
needed to forecast; Phan's "26" is the primary cohort. Phan's set is not
recoverable from the text: example patient 100 fails the rule (16 PSA and 11 androgen
points). The 43-patient set is reported descriptively.
**Pass (both):** on the 27 patients, median fit RMSE PSA ≤ 1.43 µg/L (1.25 × 1.14,
Table 2) **and** androgen ≤ 3.35 nM (1.25 × 2.68). A lower RMSE passes, since
the gate is one-sided. **Secondary (also required):** median forecast RMSE over
the next cycle, with u from the data: PSA ≤ 6.78 (1.5 × 4.52, Table 3) and androgen ≤ 6.29
(1.5 × 4.19).
**Kill.** Any of the four fails ⇒ there is no second family from this paper (ROADMAP V6
kill), and we fall back to Meade 2022 / Greene 2019 / Viossat–Noble 2021, each with its own
gate. No new tolerance, restart count, cohort or objective after a run.
**Descriptive (W5-pre lesson, precondition for step 3, not the gate).** For each fitted patient, the
day on which PSA under continuous suppression from day 0 rises above its nadir ×1.25
(confirmed 28 d later), capped at 1095 d. With no dynamic range, step 3's replay is cut.
**Artifacts:** `adaptive_therapy/envs/droop.py` (+ test),
`research/v6_phan_fit.py` → `research/outputs/v6_phan_fit.{json,log}`.

## [2026-10-04] — V7: positional-MLP baseline (POPGym's standard arm) on S1's design
**Hypothesis:** a memoryless MLP that also sees elapsed time (clock = summed dt, /357 d) can learn a time-varying threshold. POPGym (arxiv-2303.01859 p.9, 1°) found it often beats the plain MLP and calls it a standard baseline. No mechanism here argues for the clock at regular visits; under missed visits it reveals how long since the last decision. So a clock gain at Tier 4 alone would be timing, not memory.
**Arm:** `MLPPolicy(width=386, layer_norm=True, clock=True)`, Q-path 3091 vs 3090 (test). Everything else is S1's, unchanged: PQN config, Strobl ranges, w ∈ {0.05, 0.1, 0.2, 0.4}; Part A seeds 31–40, trained at p = 0, tested on Tiers 0 and 1; Part B seeds 41–50, trained at p = 0.35, tested at p ∈ {0, 0.2, 0.5}. Same seeds ⇒ same cohorts, paired against S1's saved shards. Intention-to-treat; an arm is invalid above 5% non-finite cells.
**Primary (two-sided, reported whatever the sign):** Δ(Pos, MLP) on U with seed+patient bootstrap CI: (A) Tier 1; (B) Tier 4, mean over p ∈ {0.2, 0.5}. "Clock helps" iff lower CI > 0; "hurts" iff upper < 0; else "ties".
**Secondary:** Δ(Pos, X) for X ∈ {LSTM, ODE-RNN} in A (Tiers 0, 1) and X ∈ {LSTM, LSTM+Δt, ODE-RNN} in B. Also Pos − MTD at p = 0.5. Median of paired differences beside each.
**Kill / consequence:** no kill; it is a baseline and enters the paper whatever the reading. If Pos beats the MLP in A, "the best learned policy is a memoryless threshold" becomes "a time-varying threshold". If Pos ≥ ODE-RNN at Tier 4, the continuous-time arm loses to a clock. No width, budget, seed or w changes after the run.
**Artifacts:** `adaptive_therapy/policies/mlp.py` (clock), tests; `research/v7_posmlp.py` → `research/outputs/v7_posmlp.json`, `v7_shards/`, `v7_policies/`.

## [2026-10-17] — V8: scramble the clock of V7's trained positional MLPs (no retraining)
**Hypothesis:** V7's reading (L1414) is that a clock lets PQN fit a time-indexed schedule instead of feedback, so the positional MLP loses 57 d to the MLP. If true, its trained policies depend on the clock input, and giving them a clock that carries no information must cost U. If they do not, the deficit has another cause and the paper's sentence is withdrawn. The reading is post hoc and untested until now.
**Arms (test time only):** V7's 40 saved Part-A policies (`v7_policies/a/`, w ∈ {0.05, 0.1, 0.2, 0.4} × seeds 31–40) with the clock input replaced by (i) `frozen`: 178.5 d at every visit; (ii) `reversed`: 357 d − elapsed. `intact` (the unchanged policy) must reproduce V7's shard to 1e-6 on ttp, exposure and burden integral, or the run is void. Same seeds ⇒ same cohorts and eval keys.
**Primary (two-sided, reported whatever the sign):** Δ(frozen, intact) on U at Tier 1, seed+patient bootstrap CI (w4.delta_ci). "Uses the clock as a schedule" iff upper CI < −10 d (the program's 10-d gate); "ignores it" iff CI ⊂ (−10, +10); else "inconclusive". `reversed` is a consistency check, not a gate.
**Secondary:** the same at Tier 0; median of paired differences beside each; median TTP + IQR, median exposure and burden integral (metastasis proxy) per arm; closed-loop share (observation_dependence) intact vs scrambled; paired Wilcoxon on per-patient U (frozen vs intact), pooled over seeds.
**Kill / consequence:** no kill; a control. "Schedule": the paper's reading becomes tested and moves out of "not tested". "Ignores": the sentence and the Table-V7 mechanism claim are deleted; the deficit is attributed to training only. No other scramble, no second clock value after seeing results.
**Cohort/plane:** LV, Strobl ranges, Tiers 0/1, 200 test patients, 10 seeds × 4 w; n seeds = 10. CPU, evaluation only.
**Artifacts:** `research/v8_clockscramble.py` → `research/outputs/v8_clockscramble.json`; test in `adaptive_therapy/tests/test_policies.py`.

## [2026-10-17] — W8: scheduled visits with a known next-visit time (CLOSED #2's reserved re-open condition)
**Hypothesis:** CLOSED #2 tested i.i.d. missed visits, where the gap to the next visit is unknown when a dose is chosen, so Δt describes only the past. With a fixed schedule the policy KNOWS how long its dose will be held. A continuous-time state can then flow its latent forward to that known time, and may carry across gap patterns it was not trained on; a discrete net must relearn each pattern. Prior: the MLP given the same number ties it (the mechanism is not forced); reported whatever the sign.
**Design:** LV, Strobl ranges, Tier 0, S1's PQN config, w ∈ {0.05, 0.1, 0.2, 0.4}, 52 slots, seeds 61–70, 200 test patients. Trained on schedule A = gaps (1, 3) slots (26 visits; the 50%-missed visit count). Every arm gets `lookahead` (slots to the next visit, /4) as a second observation: `mlp_L` (width 386, Q-path 3091), `lstm_L` (25 units, + past Δt, 3028), `ode_L` (width 62, 3073), all within 2% of 3091. Control `mlp_blind` (S1's MLP, width 441, no lookahead, A). Tests, no retraining: A, and held-out gap patterns B = (2, 2) and C = (1, 2, 3), same mean gap. References on the same cohorts and schedules: MTD, thr0.8.
**Primary (two-sided, reported whatever the sign):** Δ(ode_L, mlp_L) on U = min(TTP, 357) − w·exposure, mean over w, seed+patient bootstrap CI (w4.delta_ci): (P1) held-out, mean over B and C; (P2) A. "Edge" iff lower CI > 0; "deficit" iff upper < 0; else "ties".
**Manipulation check:** Δ(mlp_L, mlp_blind) at A. If its CI does not exclude 0, lookahead gives nothing to use and the run says nothing about continuous time; reported as that.
**Secondary:** Δ(ode_L, lstm_L), each arm − MTD and − thr0.8 per schedule; medians of paired differences; median TTP + IQR, exposure, burden integral; closed-loop share; paired Wilcoxon per w. Intention-to-treat; an arm is invalid above 5% non-finite cells.
**Kill / consequence:** P1 and P2 both "deficit" or "ties" ⇒ CLOSED #2 extends to scheduled visits (new item, re-open condition removed); P1 "edge" ⇒ LAW L1 gets a stated exception for known schedules and the paper's claim is narrowed accordingly. No other schedule, width, budget, seed or w after seeing results.
**Artifacts:** `adaptive_therapy/rl/rollout.py` (schedule, lookahead), `policies/{lstm,ode_rnn}.py` (in_dim), tests; `research/w8_scheduled.py` → `research/outputs/w8_scheduled.json`, `w8_shards/`, `w8_policies/`.
**AMENDMENT [2026-10-17], before any W8 cell ran:** two arithmetic corrections, no result seen. (1) `lstm_L` at 25 units is 3028 Q-path parameters, −2.0% of 3091 (26 units is +5.2%), so "within 2%" is "within 2.1%". (2) Schedule C = (1, 2, 3) has 27 visits in 52 slots, not 26 (A and B have 26): same mean gap, one extra visit.

## [2026-10-17] — W9: scheduled visits, take 2 — train on a pool of schedules so the lookahead is usable
**Hypothesis:** W8 ([L1438](RESEARCH_LOG.md#L1438)) trained on one schedule, where the lookahead is a fixed phase the MLP could ignore (check: +9.7 [−0.4, +19.4]), so it could not test whether a known next-visit time helps a continuous-time state. If patients are trained on a POOL of schedules, the next-visit time varies and must be read; a latent that flows to an arbitrary known gap may then generalise to gaps longer than any seen (D, E), where a discrete net has no training signal. This is the one mechanism left unclosed in CLOSED #2.
**Design:** W8's cohort, trainer, arms, widths (mlp_L 386, lstm_L 25 + Δt, ode_L 62) and w grid; seeds 71–80. Train pool = {(1,3), (2,2), (1,1,2), (3,1), (1,2)} (a schedule drawn per patient from the episode key; gaps ≤ 3). Control `mlp_blind` (441, no lookahead) on the same pool. Tests, no retraining: A = (1,3) (in pool); held out C = (1,2,3), D = (4,), E = (1,1,4) (D and E have gaps longer than any trained). References MTD, thr0.8 on the same schedules.
**Manipulation check (decided first, as a gate):** Δ(mlp_L, mlp_blind) on U, mean over C, D, E and w, seed+patient CI. **Usable iff lower CI > 0.** If not usable, W9 is reported as uninformative about continuous time and NO further lookahead variant is registered.
**Primary (two-sided, reported whatever the sign):** P1 Δ(ode_L, mlp_L) on U, mean over C, D, E. "Edge" iff lower CI > 0; "deficit" iff upper < 0; else "ties". **Secondary:** P1 on D, E only; Δ(ode_L, lstm_L); each arm − MTD, − thr0.8 per schedule; medians; median TTP + IQR, exposure, burden integral; closed-loop share; paired Wilcoxon per w. Intention-to-treat; an arm is invalid above 5% non-finite cells.
**Kill / consequence:** check usable and P1 ties or deficit ⇒ CLOSED #2 is extended to scheduled visits (new item, no re-open condition left). Check usable and P1 edge ⇒ LAW L1 gets a stated exception for known schedules. Check not usable ⇒ nothing is closed, and the paper reports W8 + W9 as "no test of the known-schedule mechanism was possible". No other pool, width, budget, seed or w after seeing results.
**Artifacts:** `rollout.py` (`schedule_pool`), tests; `research/w9_pool.py` → `research/outputs/w9_pool.json`, `w9_shards/`, `w9_policies/`.

## [2026-10-17] — V9: clock-dropout training of the positional MLP (does the schedule shortcut explain V7's deficit?)
**Hypothesis:** V7's positional MLP trails the MLP by 56.8 d [−64.2, −48.3] at Tier 1 ([L1414](RESEARCH_LOG.md#L1414)); V8 ([L1437](RESEARCH_LOG.md#L1437)) found scrambling its clock costs only 14.6 d [−44.2, +10.2] with seed effects from −117 to +32 d, so the clock explains part of it at most. If the deficit is the trainer fitting a clock-keyed schedule instead of feedback, a net that is trained with the clock hidden in half of the episodes must learn feedback, and a clock-shown test should recover toward the MLP.
**Arms:** `cd_mlp` = `MLPPolicy(386, layer_norm, clock_dropout=True)`, Q-path 3091 (= positional MLP). The clock is replaced by its mid-episode value in episodes whose first observation is positive (about half). Tested `show` (clock always shown; primary) and `hide` (never shown). Everything else is S1/V7's: Part A seeds 31–40, trained at p = 0, Tiers 0/1, w ∈ {0.05, 0.1, 0.2, 0.4}; paired against S1's and V7's saved shards (same seeds ⇒ same cohorts).
**Primary (two-sided, reported whatever the sign):** P1 Δ(cd_mlp[show], pos_mlp) on U at Tier 1, seed+patient CI. "Shortcut confirmed" iff lower CI > +10 d (the program's gate); "refuted" iff upper CI < +10; else "inconclusive". **Secondary:** Δ(cd_mlp[show], mlp), Δ(cd_mlp[hide], mlp), Δ(cd_mlp[hide], cd_mlp[show]) at Tiers 0/1; medians; closed-loop share (V7: pos 0.26, MLP 0.48); median TTP + IQR, exposure, burden integral; paired Wilcoxon. Intention-to-treat; invalid above 5% non-finite cells.
**Kill / consequence:** no kill; it is a mechanism test. Confirmed ⇒ the paper's schedule sentence is upgraded to a tested statement. Refuted ⇒ the sentence is withdrawn and the deficit attributed to extra-input training cost, untested. No other dropout rate, mask rule, width or budget after seeing results.
**Artifacts:** `policies/mlp.py` (`clock_dropout`, `forced`), tests; `research/v9_clockdropout.py` → `research/outputs/v9_clockdropout.json`, `v9_shards/`, `v9_policies/`.
**ERRATUM [2026-10-17], before any W9 or V9 cell ran:** the W8 log entry is at [L1457](RESEARCH_LOG.md#L1457); the "L1438" links in W9's hypothesis point to the wrong line (V8's body).

## [2026-10-17] — V10: state-dropout training of the LSTM and ODE-RNN (is the recurrent deficit V9's shortcut?)
**Hypothesis:** V9 ([L1477](RESEARCH_LOG.md#L1477)) removed the positional MLP's 57-d deficit by hiding its clock in half of the training episodes, and that network then beats the LSTM and ODE-RNN by 42–48 d under the same trainer. If the recurrent arms lose to the MLP for the same reason (PQN fits a history-keyed schedule instead of feedback), training them with their memory unavailable half the time should remove most of the −44 to −50 d. If it does not, the recurrent deficit is something else and the "learning cost exceeds benefit" reading stands against a fairer baseline.
**Arms:** `lstm_sd` = StateDropout(LSTMPolicy(26 units, LayerNorm)); `ode_sd` = StateDropout(ODERNNPolicy(width 64, hidden 16, LayerNorm)); no parameters added (Q-path 3045 / 3091 as S1). In the training episodes whose first observation is positive (about half) the recurrent state is reset to its initial value at every visit; in the rest the net is unchanged. Everything else is S1's: Part A seeds 31–40, PQN config, Strobl ranges, w ∈ {0.05, 0.1, 0.2, 0.4}, trained at p = 0, Tiers 0/1; paired against S1's saved shards. Tested `show` (memory always available; primary) and `hide` (never).
**Primary (two-sided, reported whatever the sign), Tier 1:** P1 Δ(lstm_sd[show], lstm) and P2 Δ(ode_sd[show], ode_rnn) on U, seed+patient CI. "Shortcut explains the deficit" iff lower CI > +10 d (the program's gate); "does not" iff upper CI < +10; else "inconclusive". **Secondary:** Δ(·[show], mlp) for both arms, Tiers 0/1 (edge iff lower CI > 0, deficit iff upper < 0, else ties); Δ(ode_sd[show], lstm_sd[show]); Δ(·[hide], ·[show]); medians; median TTP + IQR, exposure, burden integral; closed-loop share; paired Wilcoxon per w. Intention-to-treat; an arm is invalid above 5% non-finite cells.
**Kill / consequence:** no kill; a mechanism test. Both arms "explained" ⇒ the paper is rebuilt around the shortcut, and the title's "an ODE-RNN" claim is restated against the dropout-trained arms. ODE-RNN `show` edge over the MLP ⇒ LAW L1 gets a stated exception. Neither explained ⇒ the negative reading holds against the fairer baseline. No other dropout rule, rate, width, budget or w after seeing results.
**LAW L1 note (declared before running):** L1 rejects "a different trainer, budget or reward" without demonstrated headroom (S2). V10 changes none of these: PQN, budget and reward are S1's. It changes the training input (an augmentation), and the mechanism that justifies it is V9's confirmed shortcut, a new mechanism in the sense of the re-open rule. If the owner reads L1 more strictly, V10 is still a control on V9, not a new architecture claim.
**Artifacts:** `adaptive_therapy/policies/state_dropout.py` (+ test); `research/v10_statedropout.py` → `research/outputs/v10_statedropout.json`, `v10_shards/`, `v10_policies/`.
