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
