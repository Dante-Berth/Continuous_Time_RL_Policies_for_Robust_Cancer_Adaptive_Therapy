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

A post-hoc control (declared an AMENDMENT/selection event in PREREG.md): 1-NN
on the **single scalar** log B(365) − log B(0) scores **0.824**, already above
the gate alone; the full 52-point shape adds only +0.17. The families differ in
gross growth magnitude, so **the gate passes on scale, not mechanism**.

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

## [2026-09-22] — Phase 1 infrastructure: TTP rule fixed, sanity gate built, and PROJECT.md §12's stiffness prescription measured wrong
**Verdict:** OPEN — simulator + endpoints complete; two spec corrections recorded · **Gate:** n-a (infrastructure; no policy trained, no comparison made) · **Artifacts:** `research/phase1_cohorts.py`, `research/outputs/phase1_cohorts.json`, `research/outputs/phase1_untreated_trajectories.png`, 42 tests

### Context
Phase 1 per §11: PK, reward, LHS + sanity gate, stiffness bucketing, and
`eval/metrics.py` with the non-monotone TTP rule §6.3 requires up front.

### What was tried
TTP rule = **running-nadir + 28 d confirmation** (RECIST-style): progression
when burden exceeds the best response so far by +20% and stays there.
`ttp_from_baseline` computed alongside so rule-sensitivity stays visible. Gate
criteria **magnitude-only, never shape**, so oscillatory patients survive
(§6.3). 200 LHS patients/family.

### Results
**Three corrections, all measured rather than assumed.**

1. **§12's stiffness prescription is wrong for this simulator.** The flagged
   Hahnfeldt patients are not stiff — fixed-step dt0=0.05 agrees with dt0=0.005
   to ~1e-4. They have a *fast transient*, which makes the ADAPTIVE controller
   thrash: `Kvaerno5`+PID exceeds 200k steps and **fails even batched alone**,
   while fixed-step solves 24 in 23.5 ms; a 200-patient adaptive batch did not
   finish in 600 s. Fixed-step is the default; the flag means "re-verify at a
   finer step", not "switch solvers".
2. **My first TTP rule was wrong and a test caught it.** Running-nadir alone
   still fired at d10 on a rise-then-dormancy trajectory. Correct behaviour:
   fires *earlier* than baseline on a responder-relapse (d186 vs d277 — the
   baseline rule credits 91 unearned days) and *identically* when a tumour
   progresses before responding. Both directions tested.
3. **Untreated TTP is resolution-limited.** At weekly sampling Hahnfeldt
   progresses at 7.0 d and Kuznetsov at 14.0 d with **zero IQR** — one to two
   monitoring intervals. LV: 28.1 d [21–47], 29% censored. Gate rejected 0/200
   on all families; 71/200 Hahnfeldt carry a fast transient.

### Decision
Proceed to Phase 2 **on LV**. Discrimination must come from the treated arms:
untreated Hahnfeldt/Kuznetsov progress within one monitoring interval, so that
comparison has no dynamic range. LV is the only family where untreated patients
routinely survive the horizon.

## [2026-09-22] — Phase 2 gate blocked on CPU by transient LLVM JIT allocator failure; parked for GPU
**Verdict:** OPEN — infrastructure blocker, NOT a research result; the registered gate is unrun · **Gate:** n-a (no cell completed; 0/12 shards) · **Artifacts:** `research/phase2_gate.py`, `research/outputs/phase2_gate.log`

### Context
Phase 2's stop-or-go gate was registered [2026-09-22] and its arms built and
tested (51 tests, params matched at 3076 each). Three launch attempts all died
before any cell completed.

### What was tried
`LLVM ERROR: Unable to allocate section memory` / `allocateMappedMemory failed`,
from ~16 XLA compilation threads at once. Three mitigations, in order:
1. **Thread caps** (`OMP/OPENBLAS/MKL_NUM_THREADS=4`, Eigen off) — **no effect**;
   XLA's compilation pool does not read those variables.
2. **Subprocess isolation per cell** + JSON shards + resume — correct and kept,
   but did not by itself avoid the failure.
3. **Retry ×3 per cell** — the failure recurred faster than cells completed.

### Results
**The failure is transient fragmentation, not exhaustion.** Evidence: it fires
with **22 GB free** and no large processes; a single cell failed in isolation at
16:01 and the *identical call* succeeded at 16:04 in 4.6 s; `ulimit -v` is
unlimited. Training and evaluation each run fine in isolation at every batch
size tested (n=200 eval 0.9 s; ODE train 3 updates 8.5 s). So this is
address-space fragmentation in LLVM's JIT allocator on this CPU box, unrelated
to the research question.

**No result is claimed.** 0/12 cells completed; the registered gate stands
unrun, with its criteria and declared grid unchanged.

### Decision
**Park for GPU (available 2026-09-23).** The pathology is CPU-XLA-specific, and
the grid is minutes rather than hours on GPU. Kept: subprocess isolation, shard
resume (a completed cell is never recomputed), and retry — all are correct
regardless of backend. On GPU, also **re-measure the §12 stiffness finding**
[L88](RESEARCH_LOG.md#L88): it was measured under XLA SIMD vectorisation, which
is the execution model that warning is about, so it must not be assumed to
transfer.

## [2026-09-23] — §12 stiffness finding re-measured on GPU: transfers, but L88's mechanism is wrong — Kvaerno5 fails on every patient, adaptive Tsit5 on none
**Verdict:** OPEN — L88's *decision* (fixed-step default) confirmed on a 2nd backend; its *mechanism* corrected · **Gate:** n-a (solver diagnostic, deterministic; 200 Hahnfeldt LHS, seed 0; no policy) · **Artifacts:** `research/gpu_stiffness.py`, `research/outputs/gpu_stiffness.json`

### Context
ROADMAP forbade carrying [L88](RESEARCH_LOG.md#L88) to GPU (measured ad hoc,
CPU-XLA SIMD). PREREG [2026-09-23]; both backends in one process, same patients.

### What was tried
Registered: (i) fixed-step dt0 0.05 vs 0.005 on the 71 flagged patients; (ii)
Kvaerno5+PID at L88's 1e-6/1e-8 on the flagged batch; (iii) CPU↔GPU parity.
Diagnostic: loose tolerance 1e-4/1e-6. **Post hoc (declared AMENDMENT):**
per-patient success, `throw=False`, Kvaerno5 vs *adaptive* Tsit5, flagged vs bulk.

### Results
**Registered: all pass — transfers.** (i) 8.9e-5 on both; (ii) Kvaerno5 hits
max_steps on both; (iii) parity 5.6e-7. (RTX 4090, CUDA 12, float32.)

**Post hoc — L88's mechanism does not survive:**

| solver, tol | flagged ok | bulk ok | rejected |
|---|---|---|---|
| Kvaerno5 1e-6 | 0/71 | 0/129 | 58% |
| Kvaerno5 1e-4 | 46/71 | 69/129 | 59% |
| Tsit5 adaptive 1e-6 | 71/71 (724 steps) | 129/129 (578) | 13–18% |

1. **Adaptive stepping does not thrash; Kvaerno5 does.** Tsit5+PID at L88's own
   tolerance solves all 200 in ~600 steps.
2. **The fast-transient flag does not predict failure** — unflagged patients
   fail as often, and at 1e-4 *more* often (47% vs 35%).
3. **Not a float32 floor:** 100× looser tolerance leaves rejection at ~58%.
   Cause within Kvaerno5 is unidentified; none is claimed.

**GPU is slower here:** fixed-step n=200 rollout 1.10 s GPU vs 0.12 s CPU (9×);
Kvaerno5 batch 187 s vs 33 s. Small, sequential solves are launch-bound.

### Decision
Fixed-step stays default on both backends, for the right reason now; `gate.py`
comment corrected. ROADMAP's "minutes on GPU" is unsupported for rollouts:
Phase 2's backend is chosen by measured per-update cost, not assumed. **3-patient check preceded the amendment — selection risk
on the diagnostic only; registered criteria were untouched.**

## [2026-09-23] — L128's "transient" LLVM crash was a deterministic PPO recompile leak hitting vm.max_map_count; fixed
**Verdict:** OPEN — infrastructure fix, NOT a research result; Phase 2 gate unblocked, still unrun · **Gate:** n-a (0/12 cells; no policy compared) · **Artifacts:** `adaptive_therapy/rl/ppo.py`, `adaptive_therapy/tests/test_policies.py`

### Context
Phase 2 relaunched on CPU (GPU measured slower, [L166](RESEARCH_LOG.md#L166)).
Cell 1 died with [L128](RESEARCH_LOG.md#L128)'s `LLVM ERROR: Unable to allocate
section memory`, 119 GB available.

### What was tried
Checked kernel limits: `vm.max_map_count = 65530`. Sampled the worker's
`/proc/<pid>/maps`: **+380 mappings/s, linear**, crossing 65530 in ~3 min.
Counted XLA backend compiles vs PPO updates, cache warm: 2 updates → 3,
8 → 9, 32 → 33. **Exactly one compile per update**, ~270 mappings each.

### Results
**Cause:** `train()` collected `old_logp` with an *eager* `_batch_rollout`.
`rollout()` builds a fresh scan body per call, so the cache never hit: one
recompile per update, and its JIT code mappings never freed. 300 updates ×
~270 ≈ 81k > 65530 ⇒ every cell crashes ~update 240. **Deterministic, not
transient.** L128's mechanism (fragmentation), its evidence (a single cell
"succeeding minutes later" = a shorter run, never 300 updates) and its
retries were all wrong; the thread caps were irrelevant, as it found.

**Fix:** `collect = eqx.filter_jit(...)` built once per `train()`. After: 2
compiles and +1236 mappings per `train()` call **whatever** the update count.
Regression test fails on the old code, passes on the new; 52/52 tests.
Side effect: CPU PPO update 0.52 → 0.14 s (MLP), 0.98 → 0.37 s (ODE-RNN).

### Decision
Relaunch Phase 2 on CPU with the registered grid and criteria **unchanged** —
this touches how `old_logp` is computed (cached vs recompiled executable), not
what is computed, and no result had been seen. Subprocess isolation and retry
are kept, with comments corrected. The GPU per-update figure (17–40× slower)
is measured post-fix; L166's rollout figure had no recompiles in its timed loop.

## [2026-09-23] — Phase 2 run 1 INVALID: half the Tier-1 test cohort started above carrying capacity, and diverged rollouts scored as censored
**Verdict:** OPEN — run 1's FAIL is void, NOT a negative result; gate re-run as a declared AMENDMENT · **Gate:** n-a (12/12 cells ran; cohort invalid) · **Artifacts:** `research/outputs/phase2_gate_run1_invalid.json`, `research/phase2_gate.py`, `adaptive_therapy/eval/metrics.py`

### Context
First complete Phase 2 run on CPU after the leak fix ([L207](RESEARCH_LOG.md#L207)).
Printed FAIL: w=0.05 ΔTTP +213.5 d at exposure 306 vs 103; w=0.20 −17.5 d at
178 vs 65. Two signals said the inputs were broken before any reading of them:
untreated control 75% censored, and burden-integral median NaN in 6/12 cells.

### What was tried
Constant-dose solves of the Tier-1 cohort with no policy in the loop; per-patient
checks against the shifted K and d_D; the same with y0 re-derived from shifted K.

### Results
**Cause:** `tier1_cohort` scaled K ×0.6 *after* y0 = n_0·K was built from the
unshifted K. **86–99 of 200** test patients started above capacity, so ~60% of
untreated tumours *shrank*. LV's drug term (1 − d_D·u)·crowding turns positive
above K when d_D > 1: at u=1, **28–35/200 diverged to NaN — every one above K
with d_D > 1**. NaN never exceeds the nadir, so the TTP rule scored each as
*censored*, the best possible outcome. Re-deriving y0 from shifted K: 0 above
K, 0 non-finite. Training cohorts: 0 above K, 0 non-finite — only test was hit.

Also found: the script's PASS omitted criterion 1's exposure match; and the
registered estimator is a median over **progressors only** (84–94% censored per
cell in run 1), so it must be read beside censoring.

### Decision
Void run 1. Fixes: y0 from shifted θ; TTP **raises** on non-finite burden
(regression test); PASS requires the exposure match. Registered criteria, grid
and estimator unchanged; pooled censoring added as a secondary. PREREG
AMENDMENT pre-commits that **run 2 is final** — a result it produces is not
re-graded, whichever way it points.

## [2026-09-23] — Phase 2 STOP-OR-GO FAILED: no continuous-time edge on LV under a Tier-1 shift; ODE-RNN progresses earlier in 227–234 of ~250 untied pairs
**Verdict:** CLOSED — failed as registered (run 2, final per AMENDMENT); kill criterion ⇒ STOP · **Gate:** median TTP ODE vs MLP, w=0.05: 199.5 vs 49 d / w=0.20: 21 vs 49 d · exposure 306 vs 256 (+20%) / 178 vs 220 (−19%), never matched · burden-integral 9.9e5 vs 1.49e6 / 1.37e6 vs 1.50e6 · Wilcoxon p=3.7e-18 / 1.0e-36 · 3 seeds × 200 pts · **Artifacts:** `research/outputs/phase2_gate.json`, `research/outputs/phase2_gate.log`

### Context
Registered gate, PREREG [2026-09-22] + both AMENDMENTs; run 1 void
([L242](RESEARCH_LOG.md#L242)). Untreated Tier-1 control: 21.0 d, 22% censored.

### What was tried
Declared grid unchanged: {MLP, ODE-RNN} × w {0.05, 0.20} × seeds {0,1,2}, params
3076 each, width/LR fixed, 300 PPO updates, CPU. Median = per-seed medians.

### Results
Per-seed TTP (d) at exposure — **w=0.05** MLP 49/70/49 @ 256/278/178; ODE
56/301/200 @ 178/306/354. **w=0.20** MLP 49/49/49 @ 233/220/178; ODE 21/42/21 @
177/178/280. Censoring pooled: w=0.05 ODE 49% vs MLP 70%; w=0.20 58% vs 72%.

- **Criterion 1 fails at both weights:** exposure never within ±10%, and at
  w=0.20 ODE is −28 d. Criterion 3 (both weights) therefore fails.
- **The +150.5 d at w=0.05 is a progressor-only artifact.** Paired per patient
  (censored = horizon): ODE later 48 / **earlier 227** / tied 325; at w=0.20
  11 / **234** / 355. More ODE patients progress at all (lower censoring). The
  registered test points to MLP at both weights.
- **Second endpoint cuts the other way:** ODE burden-integral lower at both
  weights — at w=0.20 with 19% *less* drug. It holds burden down, then
  progresses earlier. Mechanism not tested here; none claimed.
- **Criterion 4, recorded not re-adjudicated:** ODE at w=0.20 ties untreated
  (21 = 21 d), which the registration calls a training bug. It is treating (58%
  vs 22% progression-free); the tie is the progressor-only median. The AMENDMENT
  allows run 3 only for a crash or non-finite rollout, so **no re-run**.
- ODE seeds split: 2 of 6 cells near-max dose (0.85, 0.98).

### Decision
**STOP, as registered.** Phase 3 (multi-ODE 2×2) does not proceed: the
curriculum cannot rescue an architecture that loses the clean case. Planned
artifact `phase2_gate.png` not produced. Program direction (CLOSED.md item,
ROADMAP strikes, Tier-4 as a *new* registration) is left to the owner.

## [2026-09-23] — Phase 2 reinterpreted: the ODE-RNN arm learned an open-loop schedule in 4/6 cells, so the gate compared feedback vs none, not discrete vs continuous
**Verdict:** OPEN — narrows [L275](RESEARCH_LOG.md#L275)'s CLOSED claim; gate stays FAILED as registered, NOT re-graded · **Gate:** n-a (post-hoc read of run 2's artifact; no new run) · **Artifacts:** `research/outputs/phase2_gate.json` (`exposure_per_patient`)

### Context
Owner asked why a continuous-time policy would lose on robustness. The
exposure 178.5 recurred in 5 of 12 cells across both arms and several seeds —
a sign of a policy whose dosing does not depend on the patient.

### What was tried
Per cell, the spread of per-patient cumulative exposure over the 200 Tier-1
test patients. A closed-loop policy doses each patient by their own burden, so
exposure varies; an observation-independent policy gives every patient one number.

### Results
| arm | exposure across 200 patients | identical |
|---|---|---|
| MLP, 6/6 cells | spans ~0–357 in every cell | 9–22% |
| ODE-RNN w0.05 s0 / s2 | 178.5 / 353.5 | **100% / 100%** |
| ODE-RNN w0.20 s0 / s1 | 176.8 / 178.5 | **100% / 100%** |
| ODE-RNN w0.20 s2 / w0.05 s1 | 280–284 / 292–313 | 91% / 32% (narrow) |

**4 of 6 ODE-RNN cells ignore the observation entirely** — one dosing schedule
for every patient — and the other 2 nearly do. Every MLP cell is closed-loop.
Phase 2 therefore measured *feedback vs a fixed schedule*; that the feedback
arm is more robust to a Tier-1 shift is expected and says nothing about
continuous time. Evidence is exposure identity only — policies were not saved,
so the dose sequences themselves cannot be inspected.

Cause **not established.** Candidates, each testable: PPO at the fixed LR/300
updates never gets gradient through obs → hidden → action; the ODE hidden state
tracks the regular 7-d clock rather than burden; entropy bonus too weak to stop
early collapse (ppo.py names constant dosing as the failure it guards against).

### Decision
L275 stands as the registered outcome: **FAIL, no re-grade.** Its CLOSED claim
is narrowed to what was shown: *this ODE-RNN under this PPO budget does not
learn a feedback policy on LV*; the architecture question is **untested, not
answered**. This matches the registration's own intent ("an arm that does not
beat untreated is not a policy … a training bug"). Any fix is a NEW
registration, diagnosing the collapse first. Save trained policies from now on.

## [2026-09-23] — Why the ODE-RNN went open-loop: default init saturates it to a fixed schedule within ~10 visits before training, and the entropy bonus had zero gradient; both fixed
**Verdict:** OPEN — diagnosis + trainer fix, not a comparison; enables Phase 2b · **Gate:** n-a (diagnostic, not pre-registered; no TTP computed by design; 2 cells × 2 arms) · **Artifacts:** `research/phase2_diagnose.py`, `research/outputs/phase2_diagnose_{old,fixed}_trainer.json`

### Context
[L312](RESEARCH_LOG.md#L312): the ODE-RNN arm was open-loop in 4/6 Phase 2 cells.
Candidates from code reading: stale ODE critic, zero-gradient entropy proxy,
re-sampled actions in the loss, un-detached advantages.

### What was tried
New `replay()` and `observation_dependence()` (0 = open-loop), tested. Both arms
retrained on cells (w0.05,s0) and (w0.20,s1) at 0/10/30/100/300 updates;
entropy (max 1.10) and dependence on Tier-1 test rollouts. Then init variants.

### Results
**Old trainer:** MLP entropy 1.05 → 0.93, dependence 0.02 → **0.23–0.25**.
ODE-RNN entropy **0.24–0.26 at update 0** → 0.07–0.24, dependence **≤0.004 → 0.000–0.003**.

**The ODE-RNN is collapsed before training.** Untrained, its entropy is 1.07 at
visit 0 and **0.00 by visit 51**: each 7-day latent flow (norm 4.2 from h=0;
10.3 over 14 d) drives h to a saturated fixed point, and a ±1 observation moves
action probabilities by only 0.02–0.04. The MLP holds 1.08 at every visit.
Scaling the latent field's output layer ×0.1 keeps it at 1.06–1.08 (×0.01
same; ×0.1 kept because gap-length dependence survives at init). No TTP seen.

**Fixes, both arms:** stored-action recurrent PPO (loss replays stored obs;
advantages/returns detached); exact entropy; critic reads the post-observation
state; ODE field output init ×0.1 (params still 3076). Each has a test that
fails on the old code.

**Fixed trainer:** MLP unchanged in kind (dependence 0.24–0.25). ODE-RNN starts
at entropy 1.10 and ends **closed-loop: 0.229 and 0.107** — the second dipped to
0.006 at update 100 before recovering, so the 0.10 threshold is not safe.

### Decision
Phase 2b registered in PREREG *before* the fixed-trainer numbers existed, with
closed-loop ≥0.10 per cell as criterion (0): an open-loop cell voids the run as a
training failure. Run it unchanged. Phase 2's FAIL is not re-graded.

## [2026-09-23] — Phase 2b INVALID as registered: 3 of 6 ODE-RNN cells still open-loop under the fixed trainer; no architecture verdict
**Verdict:** OPEN — training failure per criterion (0), NOT a comparison; ODE-RNN is fragile under PPO, MLP is not · **Gate:** n-a by registration (closed-loop MLP 6/6 at 0.24–0.31, ODE-RNN 3/6 at 0.11–0.23; open 0.011 / 0.004 / 0.038) · 3 seeds × 2 w × 200 pts · **Artifacts:** `research/outputs/phase2b_gate.json`, `research/outputs/phase2b_policies/` (12 saved)

### Context
PREREG Phase 2b [2026-09-23], registered before its run on the fixed trainer
([L353](RESEARCH_LOG.md#L353)). Criterion (0): every cell closed-loop ≥ 0.10,
else the run is a training failure with no verdict.

### What was tried
Declared grid unchanged; CPU; 21 s per MLP cell, 100 s per ODE-RNN cell.
Diagnostic cells reproduced exactly (0.229, 0.107): runs are deterministic.

### Results
**Criterion (0) fails** — ODE-RNN w0.05 s1 **0.011**, w0.05 s2 **0.004**, w0.20 s2
**0.038**. The fixes removed the collapse *at init* but not *during* training:
half the ODE seeds still converge to a patient-independent schedule. All six
MLP cells are closed-loop. **Registered outcome: INVALID. Nothing is graded.**

Descriptive only — recorded because it repeats across three runs, not as a
result: per matched patient ODE-RNN still progresses earlier in 228 vs later in
100 (w0.05) and 237 vs 76 (w0.20); one-sided p(ODE later) ≈ 1.0. The median over
progressors again points the other way (+189 d, +105 d) because ODE leaves fewer
patients censored (38% vs 67%; 47% vs 70%). ODE burden-integral again lower
(0.89–1.29e6 vs 1.31–1.54e6). Exposure matched at w0.05 (278 vs 266) for the
first time — moot in a void run.

### Decision
Per the registration: the next step is a **fresh registration with another
algorithm** (SAC-discrete or recurrent TD3), both arms, not a PPO re-tune — owner
to decide before anything is built. Standing lessons for it: keep the
closed-loop gate; make the primary TTP estimator censoring-aware, since the
progressor-only median has now inverted the paired result in three runs.

## [2026-09-23] — Phase 2c FAILED on PQN, the first VALID run: both arms closed-loop, and the ODE-RNN progresses earlier in 264/283 and 337/359 untied pairs
**Verdict:** CLOSED — failed as registered; kill ⇒ STOP: no continuous-time edge on LV Tier-1 with either trainer · **Gate:** capped median TTP (median of 3 seeds) ODE vs MLP w0.05: 357 vs 357 / w0.20: 224 vs 357 d · exposure 194 vs 280 / 194 vs 268 (−31%, −27%; unmatched) · burden-integral 1.31e6 vs 1.42e6 / 1.47e6 vs 1.42e6 · one-sided Wilcoxon p(ODE later) = 1.0 / 1.0 · 3 seeds × 200 pts · **Artifacts:** `research/outputs/phase2c_gate.json`, `research/outputs/phase2c_policies/`

### Context
PREREG Phase 2c, committed before any PQN run on the gate cohorts (8cd3a80).
PQN (arxiv-2407.04811) for BOTH arms after PPO left the ODE-RNN open-loop
([L312](RESEARCH_LOG.md#L312), [L391](RESEARCH_LOG.md#L391)). Grid unchanged.

### What was tried
Table-4 PQN, fixed; LayerNorm Q-readouts, 3090 vs 3091 params; greedy eval.
19 s per MLP cell, 130 s per ODE-RNN cell, CPU. Untreated control 21.0 d.

### Results
**(0) validity PASSES — first time in any Phase-2 run.** Closed-loop MLP
0.37–0.54, ODE-RNN 0.15–0.31; every cell ≥ 0.10.
**(2) fails at both weights:** per matched patient ODE later / earlier =
19 / **264** (w0.05) and 22 / **337** (w0.20). More ODE patients progress at all
(censored 47% vs 71%; 42% vs 68%). **(1)**: capped medians tie at the horizon
at w0.05 (357 = 357) and ODE −133 d at w0.20; exposure unmatched at both.
Per-seed capped TTP — MLP 357 in all six cells; ODE 308/357/357 and 56/224/357.

**Exposure caveat, stated not resolved:** MLP doses ~40% more (mean dose
0.72–0.78 vs ODE 0.48–0.81), and on LV more drug alone lengthens TTP, so the
registered test cannot say the ODE is worse *at equal drug*. Descriptive, post
hoc, cell-level: the two ODE cells that land at MLP-like exposure (296, 249)
are the clearly worse ones (capped 308 d, 25% censored; 56 d, 18% censored)
against MLP's 357 d at 245–282 — the gap is not only dose.

**Across trainers:** the paired direction now points to MLP in all four
Phase-2 runs (L275, L312, L391, here), under two algorithms.

### Decision
**STOP, as registered.** Written as "no continuous-time edge with either
trainer". Phase 3 does not proceed on this basis. Promotion to CLOSED.md,
ROADMAP strikes, and whether Tier 4 (irregular sampling) is registered as a
NEW hypothesis — the owner's call.

## [2026-09-24] — W1: the ODE-RNN loses on in-range patients too, so Tier 1 measured a training gap, not a robustness gap; the MLP learned a burden-threshold rule
**Verdict:** OPEN — diagnostic, not registered, no gate · **Gate:** n/a (no hypothesis); paired ODE later/earlier Tier 0: 10/164 (w0.05), 13/263 (w0.20) vs Tier 1: 19/264, 22/337 · capped median TTP Tier 0 ODE 357/357/357 and 70/357/357 vs MLP 357 in all six · exposure unchanged from L424 · 3 seeds × 200 pts · **Artifacts:** `research/w1_diagnose.py`, `research/outputs/w1_diagnose.json`, regenerated `research/outputs/phase2c_{gate.json,policies/}`

### Context
ROADMAP W1, on the 12 Phase-2c policies ([L424](RESEARCH_LOG.md#L424)), no retraining.
The artifacts were absent on this machine, so the registered grid was re-run
from HEAD on CPU: every per-patient Tier-1 TTP matches L424 bit-for-bit, and the
gate output (verdict, pairs, closed-loop values) is identical.

### What was tried
Tier 0 = the same test patients without the ×1.5/×1.5/×0.6 shift, same eval key.
Dose rule: fraction of visits whose dose equals the majority dose of its
current-observation bin (20 quantile bins); the MLP's exact greedy rule read
off an obs grid.

### Results
**Tier 0 does not rescue the ODE-RNN.** The MLP wins the paired direction
in-distribution as clearly as under shift (10/164, 13/263 untied pairs; two-sided
p 8.7e-26, 2.5e-42). The shift only moves both arms toward earlier progression
(censoring falls ~10–20 pts for both). The worst ODE cell (w0.20 s0) is already
bad at Tier 0 (capped 70 d, 32% censored).
**MLP = a burden-threshold switch.** 6/6 seeds: dose 0 when burden < ~0.80×
baseline (log-ratio −0.17 to −0.30), full dose above; ≥ 0.96 of its visits
explained by the current observation. Above ~1.8× baseline some seeds switch to
0/0.5 — a region its training cohort rarely visits.
**The ODE-RNN doses from memory** (0.62–0.93 obs-explained); the cell with the
most memory use (0.62–0.64) is the worst one.

### Decision
The paper's wording: under this trainer the ODE-RNN is **worse at control, not
less robust** — L424 is a trainability/capacity gap, and a robustness claim may
not be written from it. W3 must therefore report Tier 0 beside every Tier-1
result. The lit half of W1 (arXiv 2605.02552, Gallagher 2024 to 1°) is pending
the PDFs. Next: W2 + W3 registration.

## [2026-09-24] — W3 INVALID as registered: the LSTM and ODE-RNN arms each have one open-loop cell and the frontiers overlap < 20%; the MLP holds the horizon in all 30 cell-tier results
**Verdict:** OPEN — no H_mem / H_ct verdict (criteria 0 and 0b fail); trainability result for LSTM and ODE-RNN · **Gate:** Tier-1 frontier gap LSTM−MLP −182 d [−233, −136], overlap 16%; ODE−LSTM +22 d [−41, +158], overlap 19% (Tier 0: −113 [−143, −55] / +77 [+29, +99], 17% / 21%) · closed-loop fails: ode_rnn w0.0 s0 0.071, lstm w0.4 s1 0.098 · MLP capped median 357 d, exposure 165–294 · 3 seeds × 5 w × 200 pts × 2 tiers · **Artifacts:** `research/w3_frontier.py`, `research/outputs/w3_frontier.{json,png,log}`, `w3_shards/`, `w3_policies/`

### Context
PREREG W3 (35200d4), registered before the LSTM arm existed (474cdd4:
3045 Q-path params vs 3090). PQN, Phase-2c trainer unchanged. The ODE-RNN and
MLP cells at w 0.05/0.20 reproduce Phase 2c's per-seed Tier-1 medians exactly.

### Results
**(0) validity FAILS for two arms, by one cell each:** ODE-RNN w0.0 s0 = 0.071
(it doses 355/357 — constant maximum, the w=0 anchor risk declared in the
prereg); LSTM w0.4 s1 = 0.098. **(0b) overlap FAILS**: 16–19% on Tier 1.
So: no verdict on H_mem or H_ct, as registered. Gaps above are descriptive.
**The MLP is at the ceiling in every cell:** capped median 357 d (the horizon)
on both tiers, censoring 60–87%, obs-explained ≥ 0.94. Its frontier is flat,
so above ~50% censoring the capped median cannot rank anything — a limit of
the W2 rule on this horizon, not declared in the prereg.
**The LSTM is the least reliable arm:** 5/15 cells collapse to low dosing
(exposure 18–63) with Tier-1 TTP 77–84 d; the same seed-0 policy appears at
w 0.0 and 0.10. Frontier non-monotone (Tier 0: 105, 112, 357, 175, 357 d).
**ODE-RNN sits between:** 11/15 cells at the horizon on Tier 0, 8/15 on Tier 1.

### Decision
No re-grade: the gate is not relaxed to 0.098, and the overlap rule stands.
Descriptively, a memoryless burden threshold beats both recurrent arms under
PQN at every w; adding memory (LSTM) hurt most. Before W4 the owner must
decide (a NEW registration, never a re-run of this one): a longer horizon or
restricted-mean TTP so the MLP is not at the ceiling, and whether w=0 stays
in the sweep. W4 still proceeds as a separate hypothesis.

## [2026-09-24] — W3b INVALID again: one open-loop cell in each recurrent arm on fresh seeds; the recurrent arms fail PQN's validity gate in 4 of 54 cells, the MLP in 0 of 33
**Verdict:** OPEN — no H_mem / H_ct verdict (criterion 0 fails; H_ct overlap 1%) · **Gate:** Tier-1 RMST frontier gap LSTM−MLP −45 d [−59, −40], overlap 52%; ODE−LSTM −5 d [−15, +12], overlap 1% (Tier 0: −28 [−41, −19] / −8 [−12, +5]) · closed-loop fails: lstm w0.4 s4 0.010, ode_rnn w0.2 s3 0.011 · MLP Tier-1 RMST 291–318 d, exposure 184–287 · 3 fresh seeds × 4 w × 200 pts × 2 tiers · **Artifacts:** `research/w3_frontier.py w3b`, `research/outputs/w3b_frontier.{json,png,log}`, `w3b_shards/`, `w3b_policies/`

### Context
PREREG W3b (b76543e): W3 on seeds {3,4,5}, restricted-mean TTP (RMST) as the
frontier's y-axis, w = 0 dropped. Registered before any W3b cell ran.

### Results
**(0) validity FAILS again, one cell per recurrent arm**, at weights where W3
passed (LSTM w0.40 s4 0.010; ODE-RNN w0.20 s3 0.011). Across W3 + W3b the
recurrent arms fail the gate in 4 of 54 cells (LSTM 2/27, ODE-RNN 2/27; W3's
w=0 cells included); the MLP in 0 of 27, and 0 of 6 in Phase 2c.
**(0b)** H_mem overlap passes (34%, 52%); H_ct fails (1%): the ODE-RNN's
exposures cluster at 163–180 in 8/12 cells, the LSTM's at 172–357.
**RMST removes the ceiling as intended:** MLP Tier-1 RMST 291–318 d, never
below that in any cell; LSTM 138–283; ODE-RNN 93–265. Descriptive gap
LSTM−MLP −45 d on Tier 1, CI wholly below 0 — the MLP direction on the one
comparison whose overlap is valid, but the LSTM arm is invalid, so no verdict.
Tier-1 gaps are larger than Tier-0 for H_mem (−45 vs −28).

### Decision
No re-grade. Two registrations of the same design both lost validity to a
single recurrent cell: under PQN at this budget, recurrent Q-networks go
open-loop in ~1 in 14 cells, which is itself the trainability result the
kill clause prescribes. The next registration must decide validity per CELL
(exclude and report) or per ARM, before it runs — the owner's call. W4 is
unaffected as a hypothesis but inherits the same risk.

## [2026-09-24] — Erratum to L461: the MLP's obs-explained dose fraction is ≥ 0.94, not ≥ 0.96
**Verdict:** OPEN — correction only; no conclusion changes · **Gate:** n/a · **Artifacts:** `research/outputs/w1_diagnose.json`

L461 states the MLP's dose is "≥ 0.96" explained by the current observation.
Read back from the artifact, the 12 MLP cell-tier values span 0.939–0.993;
Tier 0 alone spans 0.957–0.993. The minimum is w0.05 s2 on Tier 1. The claim
that the MLP is a burden-threshold switch stands. VERDICTS and the arXiv
2605.02552 review now say ≥ 0.94; L461 is left as written (append-only).
Found while generating `report/2026-09-24/` from the artifacts, which is the
reason the report reads numbers rather than restating them.

## [2026-09-24] — W3c VALID: memory hurts and continuous time hurts further — MLP > LSTM > ODE-RNN on the trained objective, and the MLP only ties a one-line threshold rule
**Verdict:** CLOSED — H_mem and H_ct both FAIL on a valid run · **Gate:** Δ U (TTP − w·exposure, days) LSTM−MLP Tier 1 −82.9 [−87.7, −78.0], Tier 0 −73.5 [−78.1, −69.0]; ODE−LSTM Tier 1 −15.7 [−20.4, −10.8], Tier 0 −16.7 [−21.3, −12.3] · mean U over w (Tier 1): MLP 246.8, thr0.8 245.9, MTD 223.9, LSTM 164.0, ODE-RNN 154.5, none 97.9 · excluded 3/60 cells (LSTM 2/20, ODE 1/20, MLP 0/20) · 5 fresh seeds × 4 w × 200 pts × 2 tiers · **Artifacts:** `research/w3c_utility.py`, `research/outputs/w3c_utility.{json,png,log}`, `w3c_shards/`, `w3c_policies/`

### Context
PREREG W3c (2e5a063), approved and committed before any W3c cell ran:
per-cell validity, seeds {6..10}, primary = paired U = min(TTP, 357) − w ·
exposure averaged over w ∈ {0.05, 0.10, 0.20, 0.40}; rules descriptive.

### Results
**(0) VALID — the first valid three-arm run.** Excluded: lstm w0.1 s7,
lstm w0.2 s8, ode_rnn w0.1 s8; every arm keeps ≥ 4/5 seeds per w.
**H_mem FAILS:** the LSTM is below the MLP at every w (−55 to −120 d; one-sided
Wilcoxon p(LSTM greater) = 1 at all four). **H_ct FAILS:** the ODE-RNN is below
the LSTM overall, but not uniformly: above it at w 0.10 (+17.6, p 0.009) and
0.40 (+15.2, p 0.001), far below at 0.20 (−89.4) and ≈ at 0.05 (−6.1).
**Not a robustness effect:** Tier-1 and Tier-0 Δ are within ~10 d of each
other for both hypotheses; the gaps exist without the shift (as W1 found).
**Rules:** the 0.8 × baseline threshold ties the MLP (245.9 vs 246.8 Tier 1)
and beats both recurrent arms and MTD. The trained MLP adds nothing measurable
over the rule Gallagher 2024's RL learned. AT50 is below MTD at every w.
**Secondary frontier (RMST vs exposure):** now overlaps (38–90%) and agrees in
sign: LSTM−MLP −81 d [−99, −56]; ODE−LSTM Tier 1 −9 d [−38, +1], Tier 0 −42.

### Decision
H_ct kill fires: "no continuous-time edge over a matched recurrent arm on Tiers
0/1", and no memory edge either — on LV, burden-only, under PQN, recurrence
costs 70–80 d of the trained objective and continuous time a further ~16 d.
W4 (irregular visits) proceeds as the separate registered test. Promotion of
"continuous time gives no edge on Tiers 0/1" (L424 vs MLP + here vs LSTM) to
LAWS.md is the owner's call: two closures, same mechanism, different baselines.

## [2026-09-25] — W3c re-analysis: training variance dominates; "continuous time hurts further" does not survive resampling seeds, "memory hurts" does
**Verdict:** CLOSED — W3c's H_ct failure stands as "no edge", not "worse" · **Gate:** Tier-1 Δ U (days), patient-only CI → seed+patient CI: LSTM−MLP −82.9 [−87.7, −78.0] → [−106.8, −58.7]; ODE−LSTM −15.7 [−20.4, −10.8] → [−43.0, +21.8] (Tier 0 [−45.7, +24.2]) · per-seed ODE−LSTM −2, −47, +94, −40, −35 · one ODE cell retrained under 2e-7 rounding: mean U 132.2 → 255.5 d · 5 seeds × 4 w × 200 pts, W3c's valid cells · **not pre-registered** · **Artifacts:** `research/w3c_seed_variance.py`, `research/outputs/w3c_seed_variance.{json,log}`

### Context
Found while building W4 (commit 10ccf13). The ODE-RNN
flow's gap became a traced value, which changes Q-values and gradients by 2e-7
relative (float rounding; verified old vs new code on one batch). Retraining
W3c's `ode_rnn w0.2 s7` under that rounding gave a different policy. W3c's CIs
([L565](RESEARCH_LOG.md#L565)) resample patients with each trained policy held
fixed, so training variance never enters them.

### What was tried
(a) W3c's Δ on its own valid cells, with a second CI that resamples seed
indices (jointly across w and both arms) as well as patients.
(b) `ode_rnn` and `lstm` w0.2 s7 retrained under the current code at drop 0.

### Results
**Patient-only CIs reproduce L565 exactly**, as a check. Adding seeds:
**memory-hurts survives** (LSTM−MLP stays below 0 on both tiers, every seed
−47 to −126). **ODE−LSTM does not**: its CI spans 0 on both tiers, and one seed
of five favours the ODE-RNN by +94 d. **Training is chaotic in the recurrent
arms**: per-cell Tier-1 mean U ranges 67–264 (LSTM) and 49–273 (ODE-RNN) across
seeds at fixed w, against 177–295 for the MLP. The 2e-7 retrain moved one ODE
cell by +123 d (182/200 patients differ); the LSTM, whose path is unchanged,
retrained bit-identically.

### Decision
W3c's registered verdicts stand (H_mem and H_ct both FAIL: no CI above 0). Its
wording "continuous time hurts further (~16 d)" is withdrawn. With seeds
resampled, the ODE-RNN is **not distinguishable from the LSTM** on Tiers 0/1:
no edge, and no established deficit. Every later architecture CI resamples
seeds and patients; W4's prereg adopts this and 10 seeds. Stale downstream: the
2026-09-24 report deck states MLP > LSTM > ODE-RNN. The LAWS promotion of "no
continuous-time edge on Tiers 0/1" is unaffected in substance (still no edge),
and remains the owner's call.

## [2026-09-25] — W4 INVALID: under missed visits the recurrent arms learn near-fixed schedules, and missed visits erase the MLP's feedback edge over MTD
**Verdict:** OPEN — no H_ct4 / H_grace / H_zs verdict; all recurrent training arms fail the gate · **Gate:** excluded (closed-loop < 0.10 at p 0): MLP 0/40, LSTM 10/40, ODE-RNN 18/40, LSTM+Δt 21/40 (limit 20%); zero-shot ODE-RNN 5/20 · descriptive, seed+patient CI, mean over w & p ∈ {.2,.5}: ODE−MLP −42.8 [−76.3, −4.0], ODE−LSTM +20.0 [−17.8, +52.2], ODE−LSTM+Δt −16.7 [−50.0, +16.1]; zero-shot ODE−MLP −63.1 [−92.6, −25.9] · mean U over w, p 0 → 0.5: MLP 288 → 216, thr0.8 282 → 193, MTD 260 flat · 10 fresh seeds × 4 w × 200 pts · **Artifacts:** `research/w4_irregular.py`, `research/outputs/w4_irregular.{json,png,log}`, `w4_shards/`, `w4_policies/`

### Context
PREREG W4 (f371e5a), approved and committed before any W4 cell ran: train at
35% missed visits (hold), test Tier-0 patients at p ∈ {0, 0.2, 0.5}; seed+patient
CI ([L596](RESEARCH_LOG.md#L596)); W3c's per-cell gate at p = 0.

### Results
**(0) INVALID.** LSTM (25%), ODE-RNN (45%) and LSTM+Δt (52%) exceed the 20%
exclusion limit; only the MLP arm is valid. The zero-shot ODE-RNN is invalid too
(25%; W4's p-0 Tier-0 gate, where W3c's Tier-1 gate excluded 1/20). Per the kill
clause this is a trainability result: no architecture verdict.
**(1) The exclusions are not W3c-style failures.** 12/40 LSTM+Δt and 6/40
ODE-RNN cells sit below 0.02, i.e. near-fixed schedules, and in every recurrent
arm the excluded cells score **higher** U than the valid ones (e.g. ODE-RNN
p 0.5: 254 vs 192; LSTM+Δt 249 vs 235). Trained with unreliable observations,
recurrent Q-networks often converge on a schedule that ignores them.
**(2) Missed visits erase feedback's value.** From p 0 to 0.5, the MLP loses
72 d of U and the 0.8 × baseline rule 90 d. MTD has nothing to miss and holds
260, above both at p = 0.5 (216, 193). Mechanism under hold: a threshold
policy that switched off keeps the drug off through the gap while the tumour
regrows. The MLP still leads at p ≤ 0.2 and never drops below 200 at w ≤ 0.2.
**(3) Descriptive only, invalid arms:** LSTM+Δt valid cells are nearly flat
in p (255 → 250), and above the MLP at p = 0.5. LSTM+Δt − LSTM +14.6 [−4.7, +81.7].
The ODE-RNN is below the MLP (CI below 0) and between the two LSTMs.

### Decision
No re-grade: the gate is the registered rule, and this run cannot answer H_ct4.
The gate was built to catch collapse (L312), but at Tier 4 an open-loop
schedule can be the *better* policy, so the gate conflates the two here. A W4b
registration must first decide how validity works when observations are
unreliable. One option: keep the gate, but add the best fixed schedule (MTD) as
a reference arm every candidate must beat. Owner's call. LAW L1 is unaffected
(its scope excludes Tier 4).

## [2026-09-29] — W4b FAILED: no continuous-time edge on LV Tier 4 (missed visits), and at 50% missed a fixed MTD schedule beats every learned arm
**Verdict:** CLOSED — H_ct4b fails as registered (ODE ties MLP, loses to MTD); the negative now holds on Tiers 0, 1 and 4 · **Gate:** seed+patient CI, mean over w & p ∈ {.2,.5}: ODE−MLP −6.1 [−34.7, +12.9], ODE−LSTM +25.7 [−13.0, +56.4], ODE−LSTM+Δt +22.2 [+0.8, +43.2], ODE−MTD −23.7 [−47.5, −5.8] · capped median TTP at w 0.1, p 0.5: MLP 287 [126–357], ODE 357 [175–357], MTD 357 [357–357] · median exposure (mean over w) MLP 202, ODE 187, MTD 357 · median burden integral ODE 1.90e6 vs MTD 1.32e6 · one-sided paired Wilcoxon ODE>MLP at p 0.5: p ≤ 8e-4 at w ≤ 0.2, p = 1 at w 0.4 · 10 fresh seeds {21..30} × 4 w × 200 pts, ITT, 0 non-finite cells · **Artifacts:** `research/w4b_itt.py`, `research/outputs/w4b_itt.{json,png,log}`, `w4b_shards/`, `w4b_policies/`

### Context
PREREG W4b (4c80a0f), committed before any W4b cell ran. It fixes W4's validity
problem ([L632](RESEARCH_LOG.md#L632)) by intention-to-treat (every finite cell
counts) and adds MTD to the set the ODE-RNN must beat. **Failed as registered**;
no amendment.

### Results
**(1) H_ct4b fails.** The ODE-RNN beats LSTM+Δt (CI above 0) but only ties the
MLP and is below MTD. **H_grace fails** only against the LSTM (−3.1 [−12.8,
+6.8]); against MLP (+29.0 [+19.2, +38.4]) and LSTM+Δt (+17.4 [+0.2, +36.6]) the
ODE-RNN degrades less. Mean U over w, p 0 → 0.5: MLP 281 → 217, ODE 246 → 229,
LSTM 217 → 202, LSTM+Δt 241 → 194, thr0.8 279 → 189, MTD 258 flat.
**(2) H_fb fails: feedback stops paying at 50% missed visits.** The best learned
arm there is the ODE-RNN, still −29.1 [−55.8, −8.2] below MTD; MLP −40,
LSTM −56, LSTM+Δt −63. At p 0.2 the MLP (263) is still above MTD (258).
**(3) Replication.** W4's cells re-scored ITT (post-hoc, not a test) agree on
signs vs MLP (−15.9) and MTD (−34.4); they disagree vs LSTM+Δt (−23.1 there,
+22.2 here), and W4's LSTM+Δt − LSTM +38.8 [+5.9, +72.6] did not replicate
(+3.5 [−28.4, +35.2]). Treat LSTM+Δt effects as seed noise.
**(4) Spread.** Seed 26 alone sits −110 d ODE−MLP at p 0 (other seeds −26…+24).
ITT keeps it: training unreliability is part of the arm's cost (declared risk 5).
**(5) Open-loop is not failure here.** Cells below 0.10 closed-loop: MLP 0%,
LSTM 12%, ODE 22%, LSTM+Δt 52%, with no non-finite cells.

### Decision
The kill clause fires. There is no continuous-time edge under missed visits, and
with LAW L1 the negative holds on LV Tiers 0, 1 and 4: promoted to CLOSED #2.
No LTC or Neural-CDE rescue, and no extra seeds or rates. W5 is not required.
W6 (write-up) is next. The paper's findings about the world: at regular visits a
burden threshold suffices; as visits go missing, the ODE-RNN degrades least of
the learned arms, but the value of feedback itself collapses below a fixed schedule.

## [2026-09-29] — Pre-write-up check: only 9.8% of our LV patients lie inside Strobl 2021's published ranges; the cost of resistance is far stronger than theirs
**Verdict:** OPEN — blocks W6: every LV result so far is on provisional ranges that sampling.py says may not reach a paper · **Gate:** n/a (literature check, not pre-registered) · r_R/r_S median 0.297 [0.09–0.885], 13.7% in Strobl's 0.5–1.0 · n_0 0.30–0.85, 82% ≤ 0.75 · d_S/r_S 97.8% ≤ 0.5 · jointly 9.8% of 2000 sampled patients · **Artifacts:** Strobl Table 1 read from PMC8455086 HTML (doi 10.1158/0008-5472.CAN-20-0806); cohort check inline, `sample_cohort("lotka_volterra", 2000, seed=0)`

### Context
Starting W6, `adaptive_therapy/envs/sampling.py` states its θ ranges are
provisional and "must be replaced with sourced values … before any number
computed from them appears in a paper". Their named source, Strobl et al. 2021
(PMC8455086), was still `2°`.

### What was tried
Read Strobl's Table 1 from the PMC HTML (not the PDF, so no page locators and
the review stays `2°`). Compared it with 2000 patients from our LV sampler.

### Results
Strobl Table 1: r_S = 0.027/d fixed; r_R/r_S **0.5–1.0**; turnover d_T/r_S
0–0.5, one d_T for both populations; drug kill 1.5; n_0 **0.1–0.75**; f_R
0.001–0.1. Ours: r_S 0.015–0.045 and d_D 0.8–2.0 bracket their fixed values;
f_R matches. **The resistant growth ratio does not match:** ours has median 0.30,
so a resistance cost of ~70% against their 0–50%, and 86% of our patients fall
outside their range. n_0 runs to 0.85 against their 0.75, and our d_S, d_R are
drawn separately where they use one turnover. Jointly, 9.8% of patients are inside.

**Why this can matter for every verdict.** A strong cost of resistance slows
resistant regrowth, so simple control goes a long way. That fits what we
observed: the MLP and thr0.8 sit at the 357-d horizon with 60–87% censoring
([L496](RESEARCH_LOG.md#L496)), and MTD holds 258 d flat
([L668](RESEARCH_LOG.md#L668)). Whether the rankings (MLP ≥ recurrent arms,
no continuous-time edge) hold at Strobl's weaker costs is untested.

### Decision
No verdict is re-graded: each stands for the ranges it ran on. A paper may not
report them as "Strobl-parameterised LV". Owner's call between (a) a registered
re-run of the decisive comparisons (W3c, W4b) on Strobl's ranges, or (b) writing
up on the current ranges, declared as a synthetic high-cost-of-resistance
regime, with sampling.py's rule amended explicitly rather than ignored.

## [2026-09-29] — Erratum: U is an evaluation statistic, not "the trained objective"; the reward's survival term is a constant
**Verdict:** OPEN — correction only; no comparison changes · **Gate:** n/a · **Artifacts:** `adaptive_therapy/envs/reward.py` (`step_reward`), `adaptive_therapy/rl/rollout.py`

Found while drawing the MLP's dose rule for W6. `step_reward` pays
`survival_bonus · dt` at **every** visit, whether or not the patient has
progressed (the dataclass comment "per day alive and not progressed" is wrong),
and episodes never terminate. Over an episode that term is a constant. The
training signal is therefore −(burden above 1.2 × running nadir) − w · dose;
no term of the reward is TTP.

PREREG W3c ("Change 2") called U = min(TTP, 357) − w · exposure "the survival
and exposure terms of envs/reward.py … which every cell was trained to
maximise". That is not so: U keeps the trained exposure cost, and replaces the
burden-excess term by the TTP endpoint it approximates. Every U-based comparison
(W3c [L565](RESEARCH_LOG.md#L565), [L596](RESEARCH_LOG.md#L596); W4b
[L668](RESEARCH_LOG.md#L668); S1) stands as registered: all arms share one
reward and one statistic. Only the wording changes: "U", or "TTP net of the
exposure cost", never "the trained objective".

It also explains a feature of the learned rule. Past ~1.4× baseline (Strobl
ranges; S1's 40 MLP policies on a burden grid) the MLP steps down from full dose
to mostly half (78% half, 15–23% off). The provisional MLPs do the same past
~1.8× ([L461](RESEARCH_LOG.md#L461)). Checked on Tier-0 rollouts: 99.9% of
visits above 1.3× baseline come at or after TTP. There, TTP is fixed and a
dose only adds exposure cost.
VERDICTS and LAW L1 now say "U"; `reward.py`'s comment is corrected, code unchanged.

## [2026-09-29] — S1 FAILED as registered: on Strobl 2021's LV ranges, no continuous-time edge at Tiers 0, 1 or 4; the negative replicates across both θ regimes
**Verdict:** CLOSED — H_ct_S and H_ct4b_S fail; MLP > LSTM ≈ ODE-RNN at regular visits, and no learned arm beats MTD at 50% missed visits · **Gate:** Part A Tier 1, w 0.1: capped median TTP MLP 273 [182–357], ODE 217 [84–308]; exposure 241 / 243; burden integral 1.01e6 / 9.66e5; Wilcoxon p(ODE > MLP) = 1 at every w · Part B w 0.1, p 0.5: ODE 336 [252–357], MLP 259 [161–357], MTD 343 [273–357]; exposure 326 / 194 / 357 · 10 fresh seeds per part × 4 w × 200 pts, ITT, 0 non-finite cells · **Artifacts:** `research/s1_strobl.py`, `research/outputs/s1_strobl.{json,png,log}`, `s1_shards/`, `s1_policies/`

### Context
PREREG S1 (8e34d2b), committed before the run. One smoke cell (a mlp w0.05 s31)
ran after the prereg was written and before its commit; it is a deterministic
registered cell, resumed unchanged. Only the θ box changed ([L703](RESEARCH_LOG.md#L703)).
**Failed as registered**; no amendment.

### Results
**Part A (regular visits), seed+patient CIs.** Tier 1: ODE−MLP −50.0 [−63.5,
−37.9] (all 10 seeds < 0), ODE−LSTM −5.9 [−27.9, +19.9], LSTM−MLP −44.1
[−64.8, −30.1]. Tier 0 agrees (−50.4, −8.4, −42.0), so this is not a robustness
effect. The MLP ties thr0.8 again: mean U 215.0 vs 214.9 (Tier 1), 282.0 vs 282.8
(Tier 0). W3c's ordering on the provisional box ([L596](RESEARCH_LOG.md#L596)) replicates.
**Part B (35%-missed training).** ODE−MLP −5.4 [−24.3, +9.3], −LSTM −16.1
[−37.7, +2.3], −LSTM+Δt −1.7 [−25.2, +17.7], −MTD −13.5 [−33.0, +2.3].
H_grace fails: ODE degrades less than MLP (+38.9 [+26.7, +48.8]) but **more than
LSTM** (−20.3 [−33.9, −8.3]), so graceful degradation is a recurrence effect,
not continuous time. Mean U over w, p 0 → 0.5: MLP 281 → 206, thr0.8 284 → 159,
LSTM 233 → 247, ODE 237 → 224, LSTM+Δt 233 → 226, MTD 242 flat.
**H_fb fails, regime-dependent.** At p 0.5 the best learned arm is the LSTM,
+5.4 [−3.5, +13.6] vs MTD (a tie; W4b: every arm below). MLP −36.0 [−44, −28],
ODE −17.7 [−39.7, 0.0], LSTM+Δt −16.3 [−29.4, −4.3]. At p 0.2 the MLP still
leads MTD (262 vs 242). Where the ODE-RNN beats the collapsing MLP at p 0.5
(one-sided Wilcoxon p ≤ 0.003 at every w), it doses 326 vs 194: TTP bought with drug.
**LSTM+Δt − LSTM −14.4 [−27.0, −1.7]**, against +38.8 in W4 and +3.5 in W4b:
the Δt input's effect is not stable across seeds or regimes.

### Decision
Both kill readings fire: the negative is not an artefact of the provisional box.
The paper reports Strobl's ranges as primary and the provisional box as a
second regime. LAW L1 gains a third confirmation (a second θ regime); CLOSED #2
a second (Tier 4 on Strobl ranges). Proposed, not promoted, because promotion is
the owner's call: "recurrence costs control on LV at regular visits" (W3c, S1 A:
two regimes). The paper's Tier-4 wording becomes "no learned policy beats a
fixed maximum-dose schedule at 50% missed visits", not "every arm is below MTD".

## [2026-09-29] — S2 FAILED as registered: a reference that also sees S, R and θ scores 63 d BELOW the burden-only MLP; PQN at this budget cannot use extra state, so headroom is unmeasured
**Verdict:** OPEN — no headroom claim; the trainer, not the information, binds · **Gate:** Tier 0 Δ U O_full−MLP −63.2 [−75.5, −50.4] (all 10 seeds < 0), O_state−MLP −49.0 [−60.1, −39.4]; Tier 1 −75.4 / −55.2 · w 0.05, Tier 0: capped median TTP MLP 357 [315–357], O_state 308 [196–357], O_full 259 [175–357]; exposure 221 / 204 / 179; burden integral 1.34e6 / 1.29e6 / 1.45e6 · one-sided Wilcoxon p(O > MLP) = 1 at every w · 10 seeds × 4 w × 200 pts per part, ITT, 0 non-finite · **Artifacts:** `research/s2_headroom.py`, `research/s2_diagnose.py`, `research/outputs/s2_headroom.{json,log}`, `s2_diagnose.json`, `s2_shards/`, `s2_policies/`

### Context
PREREG S2 (3df03bf), committed before the run; one smoke cell (a o_full w0.05
s31) ran after the prereg was written, before its commit, and was resumed
unchanged. Owner-approved exception to burden-only: the reference is never an
arm. A guard re-scored S1's saved MLP cells bit-identically under the new code.
**Failed as registered**; no amendment.

### Results
**Primary, as registered:** CI wholly < 0 ⇒ "the trainer cannot use the input;
no headroom claim either way". Every seed is negative for both references
(O_full −28 to −94 d, O_state −33 to −82); they dose less and progress earlier.
**Part B's registered reading is not claimed.** O_full−MTD at p 0.5 is −55.3
[−65.7, −45.2], which the prereg reads as "collapse inherent to missed visits".
That reading presumes the reference bounds every burden-only policy; the primary
shows it does not, since it is below the MLP at regular visits. Descriptive:
O_full−LSTM at p 0.5 −60.7 [−71.4, −50.5]; O_full−MLP over p {.2, .5} −28.3.
**Diagnostic (not pre-registered; saved policies, no retraining):** the
references also earn LESS training reward than the MLP, on test patients
(O_full −15.8, higher in 5/40 cells; O_state −14.1, 4/40) and on their own
training patients (−17.7, 7/40; −17.1, 5/40). U is higher in 0/40 cells. So it
is neither reward/U misalignment (return would rise) nor overfitting (training
patients would favour it): PQN at 300 updates optimises worse with more input.
The diagnostic's test ΔU reproduces the primary (−63.16).

### Decision
Headroom stays unmeasured: under this trainer, extra state lowers U whether it
is remembered (LSTM, ODE-RNN) or given (the reference). LAW L1's "no edge"
stands on its three confirmations, but its mechanism ("the current observation
is enough") becomes an untested hypothesis. A different trainer or budget is no
longer a re-parametrization if, under it, the full-information reference beats
the burden-only MLP; only then is a continuous-time re-run worth registering.
LAWS.md amended; the paper states the mechanism as untested.

## [2026-09-30] — S3 still trainer-limited: at PQN's published parallelism the full-information reference remains 41 d below the burden-only MLP; the headroom gate stays closed
**Verdict:** CLOSED — for this paper, as registered: the budget search ends; headroom stays unmeasured · **Gate:** Tier 0 Δ U O_full−MLP −41.4 [−53.9, −27.6] (all 10 seeds < 0), Tier 1 −51.5 [−64.7, −36.8]; R (training return, O_full − MLP) −2.80, higher in 14/40 cells · w 0.05, Tier 0: capped median TTP MLP 357 [336–357], O_full 329 [203–357]; exposure 196 / 178; burden integral 1.40e6 / 1.49e6 · one-sided Wilcoxon p(O_full > MLP) = 1 at every w · 10 seeds × 4 w × 200 test pts, 2048 training pts per seed, ITT, 0 non-finite · **Artifacts:** `research/s3_budget.py`, `research/outputs/s3_budget.{json,log}`, `s3_shards/`, `s3_policies/`

### Context
PREREG S3 (9e2a71f), committed before any registered cell ran (timing only on
unregistered seed 99). LAW L1's gate ([L804](RESEARCH_LOG.md#L804)): a new
trainer or budget counts only if the full-information reference beats the MLP
under it. The one change: 1024 patients per update from a 2048-patient cohort
(was 32 from 200). **Failed as registered** (the gate did not open; reading
"still trainer-limited"); no amendment.

### Results
The registered order: CI lower bound is not > +10 d, and R < 0, so the reading
is "still trainer-limited". The budget helps the reference far more than the MLP:
O_full +29.0 [+11.3, +47.1] vs its S2 cells, MLP +7.2 [+5.3, +9.3] vs S1's
(same seeds and test patients). The gap narrows by a third (−63.2 → −41.4), and
the training-return deficit falls from −17.7 to −2.8. Both stay negative.
**Descriptive, not a test:** with the larger budget the burden-only MLP edges
past the one-line thr0.8 rule, mean U 289.2 vs 282.8 at Tier 0 and 228.4 vs
214.9 at Tier 1, by dosing less (median exposure 196 vs 238 at w 0.05). "The MLP
only ties the rule" ([L565](RESEARCH_LOG.md#L565), [L766](RESEARCH_LOG.md#L766))
holds at the 32-patient budget only.

### Decision
The kill fires as registered: no other budget in this paper. The trainer caveat
is the paper's final word on why memory does not help. The trend (a third of the
gap closed, R near 0) says a stronger trainer might open the gate. That is a
new registration for a later program (more gradient steps or another algorithm),
not a re-run here. LAW L1 is unchanged in substance; its mechanism paragraph
cites S3 as a second trainer-limited result.

## [2026-09-30] — R1 PASSED: our LV vector field reproduces Strobl 2021's published TTPs, 14/14 targets; its simulations can be built on
**Verdict:** FROZEN — Strobl's model results verified independently, not trusted · **Gate:** Fig. 4E (p.24) CT / AT TTP, want → got (d): (i) 303→296 / 323→315; (ii) 575→573 / 646→637; (iii) 616→613 / 848→844; (iv) no progression → none (both) · AT − CT gains (p.6-7), want → got: f_R 0.1% 19→19 / 211→210; f_R 1% 3→3 / 104→104; f_R 10% ≈0→0 / ≈0→0 · tolerance max(3%, 3 d): all within · deterministic, no seeds · **Artifacts:** `research/r1_strobl_repro.py`, `research/outputs/r1_strobl_repro.json`

### Context
PREREG R1 (3e998f1), committed before the run. The owner asked what we lose if
Strobl is wrong. We verified instead of trusting: `lotka_volterra.vector_field`
(no PK, u = d), Table 1 parameters, CT and AT50 schedules, and
`ttp_from_baseline` (20% over n0, no confirmation). RK4 at 20 steps per day.
**Met as registered.**

### Results
All 14 targets pass. The gains match to within a day. The absolute Fig. 4E TTPs
come out 0.5–2.5% (3–9 d) early in every case, a constant-sign offset, while
the gains are exact. That points to a difference in counting time or crossings
(their continuous-time events vs our daily grid), not in dynamics. Both
case-(iv) tumours never progress, matching the paper's indefinite control,
whose threshold (1 − 1.2 n0)·r̂_R (p.9) is 0.35 here.

### Decision
Strobl's model results are reproducible from its equations, so M1 may bin on
its cost and turnover axes. What R1 does not verify: the patient fits (Fig. 5C,
p.26). The paper keeps that sentence as a claim attributed to Strobl, never as
ours. It is also independent evidence that our LV dynamics are implemented as
specified, beyond the unit tests.

## [2026-09-30] — M1 H_cost PASSED: the missed-visit rate at which feedback stops beating MTD rises with weaker cost of resistance (0.10 → 0.26) and with turnover (→ 0.50); dosing through gaps rescues it only where resistance is cheap
**Verdict:** FROZEN — met as registered · **Gate:** Δ_fb = U(best thr) − U(MTD) at p 0.5, hold, test patients: highest-cost Strobl bin − no-cost-end bin −25.5 d [−31.2, −20.0] (low turnover), −76.6 [−88.6, −64.7] (high) · H_miss (full − hold, pooled Strobl bins) +38.5 [+36.1, +41.0] · p 0.5, hold, w 0.1, bin r̂_R [0.875,1] low turnover: capped median TTP best thr 238 [182–301] vs MTD 266 [203–322], exposure 357 / 357, burden integral 1.60e6 / 1.55e6; one-sided Wilcoxon p(thr > MTD) = 1 in every bin · 10 seeds × 10 bins × 200 selection + 200 test pts · **Artifacts:** `research/m1_monitoring.py`, `research/outputs/m1_monitoring.{json,log}`, `m1_shards/`

### Context
PREREG M1 (8a1b5ab), committed before the run: rules only, no architecture.
The best threshold x is chosen on a selection cohort and scored on a disjoint
test cohort. R1 ([L871](RESEARCH_LOG.md#L871)) verified the model first.
**Met as registered.**

### Results
**H_cost passes, same direction at both turnovers:** a stronger cost of
resistance makes missed visits favour MTD more. **p\*, the crossover
(hold):** low turnover 0.095 (r̂_R [.25,.5), outside Strobl's sweep), 0.102,
0.151, 0.189, 0.259; high turnover 0.22, 0.231, 0.266, 0.372, 0.499. It is
monotone in cost and higher at high turnover in every cost bin: feedback
tolerates missed visits longest where Strobl says adaptive therapy pays most.
**At regular visits feedback beats MTD in all 10 bins**, +13.0 to +122.6 d,
every CI > 0.
**H_miss: "hold-protocol artefact" holds on the pooled test, not uniformly.**
Under full-on-miss at p 0.5, the best rule beats MTD where resistance is
cheap (r̂_R ≥ 0.75: +5.0, +14.5, +9.2, +39.4), roughly ties at r̂_R
[.5,.75) with low turnover (−3.8, −0.6), and still loses at high cost with
high turnover (−34.3, −28.7, −20.4).
**Caveats, as declared.** (1) Horizon: in high-cost bins MTD's median TTP is
the 357-d cap with 75–100% of patients censored, so feedback can win only on
exposure there, and part of H_cost's effect is the cap. (2) Grid edge: at p 0.5
under hold the chosen x is 0.5, the lowest on the grid, in every low-turnover
bin; the best rule is drifting toward MTD. High-turnover bins choose 1.1–1.2.

### Decision
A task finding for the paper, independent of architecture: with reliable
monitoring, feedback always beats MTD. The missed-visit rate it survives spans
10–50%, set by the cost of resistance and turnover, and it depends on what is
dosed through a gap. A longer horizon would test caveat (1). It is a new
registration, not a re-run.

## [2026-09-30] — M1b: M1's cost effect survives a three-year horizon, but the monitoring picture flips at high turnover — sparing feedback beats MTD even with 60% of visits missed
**Verdict:** FROZEN — H_cost_long met as registered ("not a horizon artefact"); every absolute monitoring claim is horizon-dependent · **Gate:** p 0.5, hold, highest-cost − no-cost Strobl bin: −82.5 d [−96.6, −68.5] (low turnover), −208.9 [−255.3, −162.3] (high) · H_miss (full − hold, pooled) −5.6 [−13.9, +2.4], inconclusive (M1: +38.5) · high turnover, r̂_R [0.875,1], p 0.5, hold, w 0.1: capped median TTP best thr 196 [77–1085] vs MTD 266 [231–308], censored 29% vs 0%, exposure 304 vs 1085, burden integral 6.45e6 vs 5.33e6, one-sided Wilcoxon (U) p = 1.8e-32 · 10 seeds × 10 bins × 200 + 200 pts, 1085-d horizon · **Artifacts:** `research/m1b_horizon.py`, `research/outputs/m1b_horizon.{json,log}`, `m1b_shards/`

### Context
PREREG M1b (dfcb327): M1 with 156 weekly visits (1085 d) instead of 52, the same
patients followed three times longer. **Met as registered.**

### Results
**Registered:** the cost effect keeps its sign and grows at both turnovers, so it
is not the 357-d cap.
**The absolute picture moves with the horizon.** p\* (hold), 1 y → 3 y: low
turnover, Strobl bins 0.10→0.36, 0.15→0.42, 0.19→0.46, 0.26→0.49; high
turnover, Strobl bins 0.23–0.50 → > 0.6 in all four. At p 0.5 the best rule
beats MTD by +92.0 [+61.7, +122.9] to +300.9 [+265.2, +336.0]. The outside-sweep
high-cost bins move the other way (0.10→0.19, 0.22→0.13), because MTD holds 8% /
62% of those patients for three years.
**Why, at high turnover:** within three years MTD fails in 99–100% of Strobl-range
patients (median TTP 266–511 d). Sparing thresholds (x = 1.2) keep 29–31% of
patients controlled for the whole horizon, with ~70% less drug. Their **median**
TTP is shorter than MTD's (196 vs 266; 217 vs 511 d). The gain is a heavy right
tail plus drug saved, which U, a mean, rewards; the median does not.
**Dosing through gaps** helps at low turnover (+98 to +197 d vs hold, Strobl
bins, p 0.5) and hurts at high turnover (−135 to −188 d): pooled, inconclusive.
**Strobl's indefinite-control criterion** (38–95% of high-turnover patients) does
not transfer to our nadir rule: MTD progresses in 99–100% of them. That is
plausibly because MTD drives the nadir far down, so regrowth crosses 1.2 ×
nadir well below 1.2 n0.

### Decision
The cost effect stands. Every absolute monitoring claim is horizon-bound. That
includes W4b's and S1's "no learned arm beats MTD at 50% missed visits", which
were measured over 357 d: at high turnover, sparing feedback overtakes MTD in
years 2–3. VERDICTS, CLOSED #2 and the paper now carry the horizon. The learned
arms are not re-run: a 3-year architecture comparison would be a new
registration, and LAW L1's trainer gate still binds.

## [2026-09-30] — M1c medians: the three-year reversal holds for the median patient only where resistance is nearly free, and sparing never wins on median TTP or burden integral
**Verdict:** FROZEN — P1 mixed, P2 met at 3 y only; M1b's reversal is half a tail effect · **Gate:** M1b, high turnover, p 0.5, hold, Δ_med (median U, mean over w): r̂_R [0.5,.625) −138 [−166, −113], [.625,.75) −29 [−48, −2], [.75,.875) +34 [+4, +58], [.875,1] +62 [+37, +90] · no-cost bin, w 0.1: capped median TTP best thr 214 [105–828] vs MTD 266 [231–308], censored 24% vs 0%, exposure 378 vs 1085, burden integral 6.30e6 vs 5.33e6, one-sided Wilcoxon (U) p = 1.1e-29 · 10 seeds × 10 bins × 200 + 200 pts · **Artifacts:** `research/m1c_medians.py`, `research/outputs/m1c_medians.{json,log}`

### Context
PREREG M1c (c64705e), **post-hoc** on M1/M1b's shards: best x chosen by median U
on the selection cohort, Δ_med = median U(thr) − median U(MTD) on test patients.
Selection risk as declared: one bin's medians were seen in L932. **P1 failed its
"holds" reading as registered (mixed); P2 met at 1085 d, not at 357 d.**

### Results
**P1 (3 y, high turnover, p 0.5).** The median patient gains from sparing
feedback only in the two lowest-cost Strobl bins (r̂_R ≥ 0.75). In the two
costliest it loses (mean analysis, L932: +92 to +301 in all four). p\*_med at
3 y, high turnover: 0.35, 0.46, 0.59, > 0.6 (mean: > 0.6 in all four).
**P2 (H_cost under medians).** 3 y: −95.7 [−108.2, −73.7] (low turnover),
−200.1 [−241.4, −164.8] (high): holds. 1 y: −9.9 [−22.5, +7.6] at low turnover,
covers 0; −76.5 [−112.1, −42.0] at high. The 1-year cost effect is a mean effect
at low turnover.
**1-year p\*_med** (hold, Strobl bins): low turnover 0.30, 0.22, 0.27, 0.28 (not
monotone in cost); high 0.40, 0.39, 0.43, 0.50. M1's mean p\* was 0.10–0.26 at
low turnover: under medians, feedback survives more missed visits there.
**No bin, either horizon, has the selected threshold's median capped TTP at or
above MTD's at p 0.5** (e.g. 3 y high turnover: 210–231 vs 266–511 d). Every
median-U win is drug saved (≈ 70% less), not time bought.
**The second endpoint goes against sparing everywhere.** Median burden integral
is higher under the threshold rule in all 20 bin × horizon cells: +18% in the
3 y no-cost high-turnover bin, ~4.5× in the costliest Strobl bin at 3 y.
At p 0.5, mean-selected x scored on medians gives the same sign as median-selected x in
all 20 cells.

### Decision
M1b's "sparing feedback beats MTD at high turnover" is restated: for the median
patient only where r̂_R ≥ 0.75, only on U (exposure-weighted), with shorter
median TTP and a larger burden integral. H_cost's 1-year low-turnover leg is a
mean effect. The paper states all three. No new runs follow from this.

## [2026-09-30] — W5-pre FAILED as registered: treated Hahnfeldt and Kuznetsov have no TTP dynamic range on provisional θ; Tier 2 and cross-mechanism Stage 2 are cut
**Verdict:** CLOSED — neither non-LV family can host a dosing comparison as simulated · **Gate:** 357 d, p 0, capped median TTP [IQR]: Hahnfeldt MTD 7.0 [7, 7], every policy 7.0; Kuznetsov MTD 357 [357, 357] (100% censored) vs thr0.5–1.2 28–42 d [28–42], thr−MTD −315 to −329 d, Wilcoxon p ≤ 1e-102 / exposure 357 vs 10.5–14 / burden integral ÷ b0: Hahnfeldt MTD 19 883 vs none 23 144; Kuznetsov 3.6 vs 38–90 / 3 seeds × 200 · **Artifacts:** `research/w5_precondition.py`, `research/outputs/w5_precondition.{json,log}`

### Context
ROADMAP W5 and Stage 2 (M1's monitoring map across mechanisms) need treated
dynamic range; untreated, both families progress in 1–2 visits
([L88](#L88)). Rules only, no learning. PREREG W5-pre [2026-09-30].

### What was tried
M1's rule set (none, MTD, AT50, thr0.5–1.2) through the unchanged closed-loop
rollout, provisional θ ranges, 200 patients × seeds 1–3 per family, weekly
visits, 357 d and 1085 d, p 0 and 0.5 (hold). All trajectories finite.

### Results
**Failed as registered, both families; no amendment.**
- **Hahnfeldt: P1, P2, P3 all fail.** Every policy, MTD included, progresses at
  the first visit (7.0 d, zero IQR) at both horizons and both drop rates; MTD
  lowers the relative burden integral by only 14%. Arithmetic, not fitted: at
  V = K, dK/K = b − d·V^(2/3) − e·u, with b ≥ 3 /day and the drug term e·u ≤
  1.2 × 0.67 ≈ 0.8 /day at full weekly dosing, so no policy can stop K from growing.
  Whether this is the biology or a unit mismatch in our normalised effect (u ≤ 1)
  cannot be settled while Hahnfeldt 1999 is 2°.
- **Kuznetsov: P1 and P3 pass, P2 fails.** The outcome is set by the policy, not by
  the patient: MTD eradicates every patient (100% censored, burden integral
  3.6 b0·d), while every sparing rule progresses at 28–42 d with IQR ≤ 7 d
  (≤ 21 d at p 0.5). Kill rate u ≈ 0.67 /day is ~20× growth a ≤ 0.03 /day,
  and the family has no resistant compartment, so there is nothing for
  sparing to exploit: MTD is optimal by construction.
- Horizon (1085 d) and missed visits (p 0.5) change neither picture.

### Decision
Per the kill: both families are cut from W5 (Tier 2) and from Stage 2's
cross-mechanism monitoring map, and this is stated as a limitation. LV is the
program's only family with a resistance mechanism, so any adaptive-therapy
result is LV-only. No θ, PK/effect, TTP-rule or policy change is tried. Re-open
only with 1°-read literature ranges and a drug-effect scale (new data), plus, for
Kuznetsov, a resistance mechanism (a new model, not a re-parametrization).

## [2026-10-01] — T1 reward-limited: with 10× PQN's gradient steps the full-information reference ties the MLP on training reward but still trails it by 20 d on U
**Verdict:** OPEN — the trainer no longer binds; the reward (no TTP term) does, so headroom stays unmeasured · **Gate:** Tier 0 Δ U O_full−MLP −19.6 [−25.2, −14.7] (all 10 seeds < 0), Tier 1 −53.7 [−66.7, −42.4]; R +2.07, higher in 19/40 cells · w 0.05, Tier 0: capped median TTP MLP 357 [336–357], O_full 357 [252–357]; exposure 217 / 184; burden integral 1.42e6 / 1.61e6 · one-sided Wilcoxon p(O_full > MLP) 0.99–1.0 at w ≤ 0.2, 0.28 at w 0.4 · 10 seeds × 4 w × 200 test pts, ITT, 0 non-finite · **Artifacts:** `research/t1_steps.py`, `research/outputs/t1_steps.{json,log}`, `t1_shards/`, `t1_policies/`

### Context
PREREG T1 (191e1e3), committed before any cell ran; a follow-up program to S3
([L840](#L840)), which named "more gradient steps" as a new registration. The
one change: 3000 updates instead of 300 at 1024 patients per update. **Failed
as registered** (the gate did not open); reading "reward-limited"; no amendment.

### Results
In the registered order: the CI lower bound is not > +10 d; R is > 0, so not
"still trainer-limited"; the CI upper bound is < 0 ⇒ **reward-limited**.
- **Steps help the reference only.** O_full +22.0 [+9.7, +34.1] vs its S3
  cells, MLP +0.2 [−0.7, +1.2]. The gap halves (S3 −41.4 → −19.6) and the
  training-return deficit turns positive (S3 −2.80 → +2.07).
- **R is a tie, not a win:** higher in 19/40 cells. What holds firmly is that
  the reference now optimises the reward about as well as the MLP and still
  loses on U, at every w (−24.3 at w 0.05 to −12.0 at w 0.4) and at Tier 1.
- It doses less (184 vs 217) and carries more burden (1.61e6 vs 1.42e6) with a
  wider TTP spread (IQR 252–357 vs 336–357). The reward penalises burden above
  1.2 × nadir and dose, never TTP ([L739](#L739)), so a policy can trade TTP
  for drug at no reward cost.
- Descriptive: MLP mean U at Tier 0 316.8 / 306.8 / 286.5 / 247.5 over w vs
  thr0.8 315.0 / 303.3 / 279.9 / 233.0; at w 0.05 the MLP again only ties the rule.

### Decision
PQN is not retired: the kill fires only on "still trainer-limited". The
binding constraint is now the reward. The next admissible step is a reward with
a TTP term, under the same gate (the reference must beat the MLP by > +10 d),
as a new registration. No continuous-time re-run until then. LAW L1's mechanism
stays untested; its paragraph and the paper's trainer caveat are updated to say
the trainer, given steps, is no longer the limit and the reward is.

## [2026-10-01] — R2: Strobl 2021's Bruchovsky fits reproduce on fit quality and the cost–turnover ridge; the relapse separation only in direction (p 0.04, post hoc)
**Verdict:** OPEN — fitted patients usable for replay (kill did not fire); Strobl's relapse claim is weaker here · **Gate:** n/a (fit reproduction, no policy): F1 excluded 12 (want 10–20; Strobl 15) PASS; F2 Pearson(r̂_R, d̂_T) 0.73 (≥ 0.5; Strobl 0.77) PASS; F3 as registered (r̂_R + d̂_T) Mann-Whitney U 223, p 0.70 FAIL · post hoc, Strobl's cost + turnover: U 144, p 0.040 (Strobl U 82, p < 1%) · 67 patients, 25 restarts, deterministic · **Artifacts:** `research/r2_bruchovsky_fit.py`, `research/outputs/r2_bruchovsky_fit.{json,log}`; data gitignored

### Context
PREREG R2 (581e2f8; amendment 459bc42, before any fit: rows sorted by day).
Strobl p.6 protocol on the 67 non-metastatic Bruchovsky IADT patients with our
own `vector_field`. R1 ([L871](#L871)) verified Strobl's simulations, not its fits.
**Failed as registered** on F3. The kill (> 25 excluded) did not fire.

### Results
- **Fit quality and ridge reproduce.** 55 of 67 kept (median r² 0.77). Cost
  and turnover are anticorrelated (Pearson(cost, d̂_T) −0.73; Strobl "strong
  negative correlation … 0.77", p.10).
- **F3 was mis-registered, and that is my error.** p.10 tests the sum of
  *cost* (1 − r̂_R) and turnover; the registration wrote r̂_R + d̂_T, from an
  ambiguous line in our literature note. I read p.10 only after F3 failed, so
  the corrected test is **post hoc, with selection risk**: U 144, p 0.040, in
  Strobl's direction (median 0.78 relapse vs 0.88 non-relapse; 11 vs 44
  patients). Neither variable alone separates (p 0.92, 0.20), as in Strobl
  (p.10). The direction reproduces; the strength (p < 1%) does not.
- Descriptive: medians r̂_R 0.46, d̂_T 0.31, n0 0.63, f_R 0.013; 36% inside
  Strobl's sweep box (r̂_R ≥ 0.5, d̂_T ≤ 0.5), so 64% lie outside M1's bins;
  80% meet Strobl's indefinite-control criterion (p.9); r̂_R spread > 0.1 among
  near-best restarts in 3.6%. That is optimiser agreement, not identifiability.

### Decision
The 55 fitted patients may be used as replay patients (P1, to register next).
The literature note's "their sum" line is corrected to name cost + turnover.
The paper may say our fits reproduce Strobl's fit quality and ridge, and that
the relapse separation holds in direction only, at p 0.04 post hoc.

## [2026-10-01] — P1 MET as registered: both negatives replicate on the 55 Bruchovsky-fitted patients, but MTD never progresses there within a year, so the replay mostly tests drug use
**Verdict:** FROZEN — no continuous-time edge and no learned arm above MTD at 50% missed, on fits to real patients; low TTP power · **Gate:** H1 (p 0) ODE−MLP −44.0 [−59.5, −30.5], ODE−LSTM −2.4 [−28.1, +27.7]; H2 (p 0.5) − MTD: MLP −41.4 [−58.7, −25.3], LSTM −1.0 [−12.8, +10.0], ODE −13.2 [−33.2, +3.3], LSTM+Δt −16.2 [−34.3, −2.7] · w 0.05, p 0: capped median TTP 357 [357–357] for MLP, LSTM, ODE, MTD; censoring 0.97 / 0.78 / 0.83 / 1.00; exposure 149 / 236 / 221 / 357; burden integral 1.52e6 / 0.88e6 / 0.84e6 / 0.64e6 · one-sided Wilcoxon p(ODE > MLP) = 1 at every w · 10 seeds × 4 w × 55 fits per part, ITT, 0 non-finite · **Artifacts:** `research/p1_replay.py`, `research/outputs/p1_replay.{json,log}`, `p1_shards/`

### Context
PREREG P1 (d85fa4a), committed before any replay. S1's 280 saved policies and
the rules, no retraining, on R2's fits ([L1075](#L1075)). The first launch
died of the L207 JIT-mapping leak (one process, 280 policies); re-run as one
process per (part, arch), with no change to the statistic. **Met as registered.**

### Results
- **H1: no edge replicates.** ODE−MLP and ODE−LSTM match S1's Tier 0 on
  simulated Strobl patients (−50.4 [−65.7, −36.2], −8.4 [−35.8, +25.0]).
- **H2: no learned arm beats MTD at 50% missed.** ODE−MTD −13.2 [−33.2, +3.3]
  vs S1's −13.5 [−33.0, +2.3]. Best is LSTM, a tie, as in S1 (+5.4). The
  patient-level Wilcoxon favours LSTM over MTD at some w (min p 5e-31); the
  registered seed+patient CI covers 0.
- **In Strobl's box (20 fits):** same readings (ODE−MLP −38.0 [−49.2, −27.4],
  LSTM−MTD +6.8 [−5.4, +18.6]). MLP ties thr0.8 (−2.3 [−6.0, +1.4]).
- **Low TTP power.** MTD keeps all 55 fits progression-free over 357 d, and
  every learned arm's median TTP is the cap, so U differences come mostly from
  drug and from the 3–22% of patients who progress. AT50 progresses early
  (median 126 d [105–175]) because our nadir rule fires on its planned regrowth.
  This matches R2: 80% of fits meet Strobl's indefinite-control criterion.

### Decision
The paper may say both negatives replicate on models fitted to 55 real
patients' PSA, with the caveat that MTD controls every fitted patient for a
year there, so the replay tests drug use more than time to progression. No
clinical claim. No further replay runs for this paper.

## [2026-10-02] — V1: the memory arms lose the training reward too, so their U deficit is not reward misalignment; "no learned arm beats MTD at 50% missed" holds only for the mean over one year
**Verdict:** FROZEN — V1a: not misalignment; V1b FAILED as registered (medians favour the learned arms at p 0.5); V1c: every learned arm beats MTD on U over 3 y, by drug saved · **Gate:** V1c, 1085 d, p 0.5, w 0.1: capped median TTP LSTM 357 [266–455], ODE 336 [252–427], MLP 259 [161–406], MTD 343 [273–434]; exposure 783 / 847 / 611 / 1085; burden integral 5.39e6 / 5.39e6 / 5.80e6 / 5.09e6 · Wilcoxon not run (registered CIs only) · 10 seeds × 4 w × 200 pts, S1's saved policies, ITT, 0 non-finite · **Artifacts:** `research/v1_review.py`, `research/outputs/v1_review.{json,log}`, `v1_shards/`

### Context
PREREG V1 (2268849), post hoc on S1's policies, registered before any number.
Three reviewer objections: a reward with no TTP term, a mean statistic, and a
one-year horizon. Reproduction guard passed: V1b's means equal S1's logged Δs.

### Results
**V1a — training return (own reward, own training cohort), X − MLP, seed CI.**
LSTM −20.7 [−28.4, −12.8] (higher in 6/40 cells), ODE-RNN −23.5 [−34.7, −14.0]
(5/40): **loses the reward too, not misalignment.** The memory arms optimise
the MLP's reward worse; they do not win it and lose U. ODE − LSTM ties (−2.9
[−16.3, +8.8]). Part B (drop 0.35, descriptive): every recurrent arm earns
less reward than the MLP, yet the LSTM earns more U on its training cohort
(+11.4 [+3.6, +19.8]): under missed visits reward and U part ways.
**V1b — median of paired ΔU.** Part A, all six contrasts agree, about half the
mean's size (Tier 1 ODE−MLP −29.7 [−48.2, −16.8]; ODE−LSTM +2.5 [−13.8, +17.9]).
Part B at p 0.5 against MTD: MLP +0.1 [−15.4, +5.5], **LSTM +10.7 [+1.7, +28.0]**,
ODE +6.9 [−9.5, +17.2], **LSTM+Δt +4.2 [+1.5, +8.5] (mean −16.3: reversed)**.
**Failed as registered:** for the median patient the LSTM beats MTD at 50%
missed. 43% of patients are censored under MTD and 47% under LSTM at 357 d.
Where both arms are censored, ΔU is drug saved. The mean is pulled down by
the minority who progress earlier.
**V1c — 1085-d replay, arm − MTD at p 0.5 (mean U).** MLP +42.5 [+27.3, +58.1],
LSTM +81.6 [+53.4, +108.5], ODE +49.5 [+19.7, +77.3], LSTM+Δt +32.0 [+13.0, +51.6].
On capped TTP alone (descriptive, same CI): LSTM +9.0 [−9.4, +28.6]; MLP −57.9
[−73.0, −43.2], ODE −43.5 [−72.9, −20.1], LSTM+Δt −48.2 [−66.4, −32.1]. Only
the LSTM buys MTD's time; the other gains are drug. ODE−MLP +7.0 [−24.0, +39.2],
ODE−LSTM −32.1 [−70.9, +9.1]: no continuous-time edge at 3 y.

### Decision
Objection 1 is answered for the architecture ablation: the memory arms lose on
the reward they were trained on, so their deficit is optimisation, not a broken
proxy. The reward still limits the headroom question, which is V2. The paper's
MTD sentence becomes: below MTD on mean U over one year; the LSTM above it for
the median patient; every arm above it on U over three years, mostly through
drug saved. CLOSED #2 is untouched. Declared risk (1) applies to V1c: the
policies run 3× past their training horizon.

## [2026-10-02] — V5 FAILED as registered: Hahnfeldt 1999 reproduces in its own drug units to within 3–10%, so W5-pre's "drug cannot outpace growth" was our scaling; Zhang 2017 does not reproduce from its main text
**Verdict:** CLOSED — V5a misses 2 of 5 targets narrowly (Hahnfeldt stays closed for dosing); V5c misses by ~7× (no second-family claim) · **Gate:** n/a (model reproduction, no policy; deterministic): V5a H1 17,347 vs 17,400 mm³, H2 11,918 vs 12,300 (−3.1%, tol 3%), H3 236 vs 240, H4 2033/1431 vs 1840/1300 (+10.5%, tol 10%), H5 endostatin 20 → 100 mm³, 4 → 2414, TNP-470 → 2657 at day 15 (pass) · V5c Patient #2: 5% T− day 27 vs 323, 90% T− under MTD day 104 vs 784, adaptive reaches 90% at 116 vs never · **Artifacts:** `research/v5_repro.py`, `research/outputs/v5_repro.{json,log}`, `adaptive_therapy/envs/zhang2017.py`

### Context
PREREG V5 (cd55e4d), before any simulation. Both PDFs read `1°` today.
Hahnfeldt: our `vector_field` unchanged, its drug term driven by the paper's
bolus PK in mg/kg (Eq. C3, p.4773; Table 1, p.4771). Zhang: Eq. 1–2, Table 3
(p.7–8), new `envs/zhang2017.py`.

### Results
**V5a — failed as registered, narrowly.** Three targets pass: the untreated set
point (analytic (b/d)^1.5 = 17,347), angiostatin 14 mg/kg/day (overshoot to 301,
then 236), and the dose response of H5. H2 misses by 3.1%, but is the same
from V0 = 170, 12,300 and 17,300, as p.4774 states. Post hoc, descriptive:
sampling 200 points per interval leaves it at 11,916, so the miss is real and
small. H4's bolus/continuous ratio is 1.42 vs the paper's 1.415. Post hoc: from
V0 = 170 (the TNP-470 start of Fig. 4 II, p.4774) day 13 is 1833/1289, within
1%. The registered V0 was p.4772's "≈200".
**What it settles anyway.** In the paper's units endostatin 20 mg/kg/day gives
e·g ≈ 0.66 × 12 ≈ 8 /day and regresses the tumour (H5). W5-pre capped the
drug term at e·u ≤ 0.8 /day through our normalised Emax PK ([L1004](#L1004)).
Its "the drug cannot outpace vascular growth" was our scaling, not the model.
The model is a mouse one: days, daily boli, set points in weeks.
**V5c — failed.** Patient #2's equilibrium has 2.4% T− (330 of 13,861 cells),
which reaches 5% by day 27, before treatment starts (day 97). Under MTD, T−
is at 90% by day 104: growing at r = 0.0665 /day it needs about ln(110)/r ≈ 70 d,
and the paper's 461 d is about 7× slower. Patient #1 (T− = 1.7e-128 at
equilibrium, a numerical seed; declared not reproducible) lands near Table 1
by accident of that seed: MTD 90% at 4641 vs 4621, adaptive 5% at 8592 vs 8557.
The main text does not fix the time unit, the seeding of T−, or how MTD
starts. The Supplementary Information, not obtained, may.

### Decision
Hahnfeldt stays closed for dosing (CLOSED #3), but the paper may now say the
W5-pre failure traced to our drug scaling, with H5 as evidence. Zhang 2017
gives no second-family claim from its main text. Re-open only with its
Supplementary Information (new data), under a fresh registration; V5's
tolerances and targets do not move. `envs/zhang2017.py` stays as the
main-text transcription.

## [2026-10-02] — V5d FAILED as registered at its last structural check: our Zhang model reproduces all 22 SI equilibria within 1.9 pp, but the paper's own Table 3 Patient #1 matrix is SI row 22, not row 7
**Verdict:** OPEN — Part 2 (time readings) not run, as registered; the failing check exposes a paper-internal inconsistency, not a transcription error · **Gate:** n/a (model reproduction, deterministic): S1 22 orderings under p.7's inequalities (want 22) PASS; S2 all 22 equilibria vs Supp. Table 1 within 1.9 pp (tol 2) PASS; Table 3 matrices → rows {Patient #1: 22, Patient #2: 5} (want 7, 5) FAIL · **Artifacts:** `research/v5d_zhang_si.py`, `research/outputs/v5d_zhang_si.{json,log}`

### Context
PREREG V5d (92dffce), registered after the SI arrived and before any V5d run.
SI on disk as `literature/pdf/url-zhang-2017-mcrpc-at-SI.pdf`.

### Results
- **S1.** Assigning {0.4, …, 0.9} to the six α under the six inequalities of
  p.7 gives exactly 22 matrices, the paper's count (p.7). Row i, column j is
  the paper's reading.
- **S2, multiset.** Our untreated equilibria match Supp. Table 1's
  k = [1.5 TP, 1e4, 1e4] column row by row (optimal assignment). Worst cell
  1.9 pp (row 12: 49.9/50.0/0.1 vs 48/50/2); rows with T− > 0 match within
  0.6 pp. The transcribed model is the paper's model.
- **S2, Table 3.** Patient #2's matrix (Table 3, p.8) gives 45.2/52.4/2.4 →
  row 5, as the SI states (p.3). Patient #1's printed matrix gives 44.4/55.6/0
  → row 22, which never progresses in Supp. Table 3. Row 7 (52/48/0), whose
  times Table 1 reports for Patient #1, needs α(T+, TP) ≈ 0.4: K_T+ = 1.5 x_TP
  gives T+/TP = 1.5 − α_12, and 52/48 means 1.1. Patient #1's printed α_12 is 0.7.
  Either Table 3's Patient #1 matrix or the SI's "row 7" label is wrong. We
  cannot tell which.

### Decision
As registered: the check failed, so Part 2 is not run under V5d, and no
Zhang simulator is adopted. The check was meant to catch a wrong transcription,
and the multiset match shows the transcription is right. Running Part 2 with
rows assigned by the SI equilibria (Patient #1 is not among Part 2's rows
anyway) would change the gate after seeing Part 1. That is the owner's call, as
a fresh registration that declares this. LV stays the only family with
resistance.

## [2026-10-02] — V5e CLOSED: no declared reading reproduces Zhang 2017's progression times (best 15/40); the structure reproduces, the times do not, so LV stays the only family with resistance
**Verdict:** CLOSED — "not reproducible from paper + SI", as registered; no second-family claim · **Gate:** n/a (model reproduction, deterministic): hits within max(3%, 3) of Supp. Table 3, 40 cells, pass ≥ 36: R-a (ODE, 0.1×) 14, R-b (ODE, 0.01×) 15, R-c (map, 0.1×) 14, R-d (map, 0.01×) 15 · **Artifacts:** `research/v5d_zhang_si.py --v5e`, `research/outputs/v5e_zhang_times.{json,log}`

### Context
PREREG V5e (7668890): V5d's Part 2, gated on S1 and the 22-row equilibrium
match only. That gate was decided after V5d ([L1216](#L1216)) found the paper's
Table 3 / SI inconsistency; owner-approved.

### Results
- **Every reading fails.** 10 of the 14–15 hits per reading are the trivial
  cells where T− is already ≥ 5% at the start (the paper's "1", our 0). Under the
  0.1× readings (R-a, R-c), MTD reaches 90% T− 6–8× sooner than the paper
  (row 5: 104 / 99 vs 784). R-b and R-d each also hit row 6's adaptive 90% T−
  (1584 / 1573 vs 1596).
- **No single rescaling would fit (descriptive, no fifth reading run).** Under
  R-b, MTD's 90% times are a steady 1.27–1.39× the paper's on all ten rows. The
  5% times go the other way on some rows (row 12: 818 vs 673, ×1.22; rows 5
  and 2: 261 vs 323 and 124 vs 161, ×0.8). Adaptive 90% runs short or long by row
  (row 12: 2119 vs 4126). The paper's simulator differs from Eq. 1–2 in some
  unstated way beyond the time unit.
- Row 5 (Patient #2), adaptive: the paper says T− never reaches 90%; every
  reading reaches it (R-b at 2529).

### Decision
Zhang 2017 is closed as a simulator for this program. Its structure
reproduces (22/22 equilibria, [L1216](#L1216)), its published times do not under
any declared reading. Re-open only with the authors' code or a statement of the
integration scheme, the time unit and the T− seeding (new data). The paper's
"LV-only" limitation stands. It may now say that a second resistance model was
attempted, and why it could not be used.

## [2026-10-02] — O1 headroom: a full-information search beats the burden-only MLP by 23 d with no trainer involved, so the task has room the MLP does not reach
**Verdict:** FROZEN — met as registered, reading "headroom" · **Gate:** Tier 0, Strobl ranges, Δ_O = U(oracle) − U(MLP) +23.1 d [+20.3, +25.8] (per w 8.6 / 16.1 / 23.2 / 44.5); tuned threshold − MLP +26.0 [+23.5, +28.5] · w 0.05: capped median TTP oracle 336 [336–357], tuned 357 [357–357], MLP 357 [315–357]; censored 24% / 65% / 67%; exposure 122 / 161 / 220; burden integral 1.98e6 / 1.95e6 / 1.34e6 · Wilcoxon not run (registered CIs only) · 10 seeds × 200 pts × 4 w, deterministic search · **Artifacts:** `research/o1_oracle.py`, `research/outputs/o1_oracle.{json,log}`, `o1_shards/`

### Context
PREREG O1 (afa0aa7). At regular visits the simulator is deterministic given the
doses, so a per-patient dose-sequence search on the true model bounds the
full-information optimum from below, with no RL trainer.

### Results
- **Headroom, as registered.** The oracle beats S1's MLP on the same patients
  by +23.1 d, lower bound well above the +10 d margin. 51% of patient-cells
  gain more than 10 d.
- **The search lost to its own warm start.** oracle − tuned −2.9 [−3.7, −2.0];
  in 54.5% of cells the returned sequence scores lower on U than (i). Declared
  risk (2) fired: the online objective cannot see progression in the last 28 d.
  61% of oracle patients progress there (median TTP 336), and the search
  traded that for drug. The tuned threshold (i) is the tighter bound.
- **Tuned threshold vs MLP (descriptive).** +14 to +19 d of capped TTP *and*
  46–60 d less drug per w. Time is bought, not only drug saved. The chosen
  levels cluster at 1.15 × baseline (IQR 1.05–1.20, every w), against the MLP's
  learned switch near 0.8 ([L739](#L739)). They carry a larger burden integral
  (1.95e6 vs 1.34e6), which the training reward charges for.
- The search had converged: +8.7 mean objective over the warm start, +0.01 over
  its last 5 iterations.

### Decision
The MLP's limit is not the task. Whether the room needs patient-specific
information (something memory could recover) or only a higher fixed threshold
is O1b, registered after seeing the level cluster. LAW L1's mechanism
paragraph is amended. No continuous-time re-run follows from this entry.

## [2026-10-02] — O1b: O1's 26 d of headroom splits into thirds — a better fixed threshold the MLP could have learned (11 d), exact observation (9 d), patient-specific information (7 d); at most ~15 d was ever available to memory
**Verdict:** FROZEN — registered reading fired ("fixed threshold, not information") but its premise was flawed; the post-hoc noisy split is the account · **Gate:** Tier 0, seed+patient CI: Δ_G (noise-free level 1.05, LOSO) − MLP +19.4 d [+17.0, +21.9]; Δ_I tuned − global +6.6 [+6.0, +7.2]; same level under S1's 5% noise − MLP +2.6 [−2.1, +7.0] · w 0.05: capped median TTP 357 for all; censored global 82% / noisy 73% / MLP 67%; exposure 191 / 192 / 220; burden integral 1.79e6 / 1.81e6 / 1.34e6 · Wilcoxon not run · 10 seeds × 200 pts × 4 w · **Artifacts:** `research/o1b_split.py`, `research/outputs/o1b_split.{json,log}`, `o1b_shards/`

### Context
PREREG O1b (11be513), registered after O1's level cluster ([L1279](#L1279)) was
seen. Leave-one-seed-out global threshold vs O1's tuned rule and S1's MLP.

### Results
- **As registered:** Δ_I's upper bound < 10 and Δ_G's lower bound > 10 ⇒ "the
  headroom is a fixed threshold PQN's reward did not reach". Level 1.05 at
  every w.
- **The premise was wrong, and that is my error.** I called the noise-free
  global rule "burden-only, memoryless, no information". It is memoryless, but it
  sees the burden without the 5% noise every arm faces. Under S1's noise the
  same level is only +2.6 [−2.1, +7.0] above the MLP (descriptive, as
  registered).
- **Post hoc, descriptive:** the best *noisy* fixed level (1.0; 0.95–1.0 at w
  0.05), chosen leave-one-seed-out on noisy scores: − MLP **+10.8 [+8.1, +13.3]**.
  Clean global − noisy best **+8.6 [+7.2, +10.5]**. With Δ_I +6.6, the three
  parts sum to O1's tuned − MLP (+26.0).
- **Reading of the split.** 11 d is a memoryless threshold near 1.0× baseline
  that the MLP could represent but PQN's reward does not lead to: its burden
  term charges for burden between the nadir line and baseline. 9 d is the value
  of exact observation, an upper bound on what filtering (memory) could recover
  from 5% noise. 7 d is patient-specific information, an upper bound on what
  inferring the patient from history could add.

### Decision
Memory had at most ~15 d to gain (9 + 7, both upper bounds); recurrent training
cost 40–50 d against the MLP (S1, [L766](#L766)). The negative is a learning
cost that exceeds the available benefit, not "nothing to integrate". LAW L1's
mechanism paragraph is rewritten. The paper's "why memory does not help"
paragraph is rewritten with this split, post hoc parts labelled.

## [2026-10-03] — V2 still trainer-limited: a survival term that stops at progression closes the reference's gap to the MLP (−19.6 → −1.4 d) but not past it, and the reference earns less of the new reward
**Verdict:** CLOSED — no headroom over the MLP from PQN under a TTP-bearing reward; the reward search ends, as registered · **Gate:** Tier 0 Δ U O_full−MLP −1.4 [−4.1, +1.2] (patients-only [−2.6, −0.2]; 6/10 seeds < 0), Tier 1 −39.8 [−49.9, −30.5]; R −5.02, higher in 11/40 cells · w 0.05, Tier 0: capped median TTP MLP 357 [322–357], O_full 357 [287–357]; exposure 225 / 187; burden integral 1.37e6 / 1.60e6 · one-sided Wilcoxon p(O_full > MLP) ≤ 1e-16 every w (descriptive; per-w mean Δ −7.7, −5.6, −1.2, +9.0) · 10 seeds × 4 w × 200 test pts, ITT, 0 non-finite · **Artifacts:** `research/v2_ttp_reward.py`, `research/outputs/v2_ttp_reward.{json,log}`, `v2_shards/`, `v2_policies/`

### Context
PREREG V2 (9858026), committed before any cell ran. The one change from T1
([L1042](#L1042)): `RewardConfig(survival_until="progression")`, the survival
term paid until online-confirmed progression. Readings in registered order.

### Results
- Primary: lo −4.1 < +10 ⇒ no headroom gate; R < 0 ⇒ **trainer-limited**
  (reading 2 fires before "no headroom", as registered).
- Reward effect vs each arm's own T1 cell (same seeds, patients): O_full
  **+15.4 [+10.2, +20.8]**, MLP −2.9 [−3.9, −2.0]. The TTP term helps the
  reference only, as the prior said; it still doses less (187 vs 225 d) and
  carries more burden.
- Per w the sign turns: O_full ahead at w 0.4 (+9.0), behind at w ≤ 0.2.
- Tier 1 unchanged in kind: reference −40 d, the T1 −54 d narrowed.
- MLP vs thr0.8 at w 0.05: 314.1 vs 315.0 mean U, less drug (225 vs 238).
- **Mean and median disagree** (post hoc, pooled paired U): O_full beats the
  MLP on 62–69% of patients, median Δ +1.7 / +2.6 / +7.0 / +15.4 d by w, mostly
  less drug; a tail of earlier progressions (p10 ≈ −30 d) makes the mean
  negative. The registered primary is mean U, so the verdict stands.

### Decision
Failed as registered (gate did not open). Per the kill clause: no third reward
and no reweighting of burden or exposure terms in this paper; V3 is not
registered. Read with O1/O1b ([L1279](#L1279), [L1310](#L1310)): the true-state
optimum is 23–26 d above the MLP, yet PQN with the true state sits at the MLP
under both rewards. The gap between what the state is worth and what PQN extracts
from it is the learning cost L1 names; V2 adds that a TTP-bearing reward does
not remove it. LAW L1's mechanism paragraph gets one sentence; the paper's
`\pending{V2}` is replaced with this result.

## [2026-10-04] — V6-gate FAILED: within Phan 2019's Table 1 ranges our two-population Droop model fits our Bruchovsky patients 3× worse than reported, with most parameters pinned to a bound
**Verdict:** CLOSED — no second family from Phan 2019; every result stays LV-only · **Gate:** 27 pts, median [Q1, Q3] RMSE: fit PSA 3.67 [3.16, 5.25] (≤ 1.43; Phan 1.14), fit androgen 5.07 [4.51, 5.64] (≤ 3.35; 2.68), forecast PSA 19.0 [15.9, 22.1] (≤ 6.78; 4.52), forecast androgen 6.87 [5.88, 7.96] (≤ 6.29; 4.19); 4/4 fail · deterministic fit, 30 restarts × 43 pts, 30/30 finite each · no Wilcoxon (no policy comparison) · **Artifacts:** `research/v6_phan_fit.py`, `research/v6b_optimizer_check.py`, `research/outputs/v6_phan_fit.{json,log}`, `v6b_optimizer_check.{json,log}`

### Context
PREREG V6-gate (7792568), committed before any fit. Phan 2019 read `1°`
(fd3976c). Model `adaptive_therapy/envs/droop.py`, eqs. 2.1–2.6; objective
MSE_PSA + MSE_androgen over on/off/on; cohort = Phan's rule (43/72) ∩ a
further full cycle (27). Declared deviations: seeded restarts instead of
hand-chosen guesses; lower bounds 0 → 1e-3 on R1, R2, K.

### Results
- All four criteria fail, by 1.5–4× the threshold. The 43-patient rule cohort
  is no better (fit PSA 3.47, androgen 5.69).
- **Not the optimiser (V6b, post hoc, not pre-registered).** Six patients
  (Phan's 28, 29, 77; 17, 25, 101) re-started from the V6 optimum with 10× the
  evaluation budget and tolerance 1e-12: every run converged (ftol) to the V6
  objective within 0.002%.
- **The optimum sits on the box:** 13–15 of 18 parameters at a Table 1 bound
  in all six. The data push past Phan's ranges, or our transcription of the
  ranges/units differs from what they fitted with. The PDF cannot separate
  the two; the invariant tests (Prop. 1 bound, no AI without λ, CAS relapse) pass.
- Patient 28, Phan's best-looking example (Fig. 2): fit PSA 10.6 here.
- Descriptive: every fitted patient relapses under continuous suppression
  (median TTP 717 d [633, 789], 0 censored). So the family would have had
  dynamic range; the gate, not W5-pre's failure mode, kills it.

### Decision
Failed as registered. Per the kill clause: no wider box, no new objective,
restart count, cohort or tolerance on this paper. That would be a
re-parametrization chosen after seeing the result. CLOSED #4 is extended to
Phan 2019. A pattern note, **not a law**: two published validations (Zhang
2017, Phan 2019) failed to reproduce from the paper alone, for different stated
reasons. A LAW needs the same mechanism twice, and here it is unproven.
ROADMAP V6 struck. The fallbacks (Meade 2022; Greene 2019; Viossat & Noble
2021) need the owner's go-ahead, each with its own gate. The paper's
LV-only scope sentence stands.

## [2026-10-04] — V7: a clock alone costs as much as memory: the positional MLP trails the MLP by 57 d at Tier 1 and ties it at Tier 4
**Verdict:** OPEN — the recurrent arms' deficit is not specific to memory: one clock input costs the MLP −56.8 d, as much as an LSTM's state (−44.1, S1) · **Gate:** Tier-1 median capped TTP Pos 175 [126, 259] vs MLP 273 [182, 357]; median exposure 212 vs 245 d; burden integral not computed; Wilcoxon not run (CI rule as registered); 10 seeds × 4 w × 200 patients per part · **Artifacts:** `research/v7_posmlp.py`, `research/outputs/v7_posmlp.{json,log}`, `v7_shards/`, `v7_policies/`
### Context
PREREG V7 [2026-10-04]. POPGym (1°, p.9) calls the positional MLP (memoryless + elapsed time) a standard memory baseline. Same seeds, cohorts, trainer and w grid as S1 ([L766](RESEARCH_LOG.md#L766)), paired against S1's shards. Width 386, Q-path 3091 vs 3090. One smoke-test cell (A, w 0.1, s31) ran before launch, with registered settings, and is kept.
### Results (Δ U, d, mean over w; 95% CI over seeds + patients; median of paired differences after)
| Part | vs | mean Δ | median Δ |
|---|---|---|---|
| A Tier 1 | MLP | **−56.8 [−64.2, −48.3]** | −44.6 [−55.4, −32.3] |
| A Tier 1 | LSTM / ODE-RNN | −12.7 [−30.0, +10.3] / −6.8 [−24.2, +12.7] | −7.2 / −6.1 (CIs cover 0) |
| A Tier 0 | MLP | −53.0 [−65.4, −40.9] | −33.5 [−53.7, −20.3] |
| B p∈{.2,.5} | MLP | **+2.2 [−13.2, +16.4]** | −7.2 [−17.9, +3.1] |
| B p∈{.2,.5} | LSTM / LSTM+Δt / ODE | −8.5 / +5.9 / +7.6 (all CIs cover 0) | −5.2 / +9.7 / +0.3 |
| B p=0.5 | MTD | −6.0 [−18.9, +4.8] | **+12.7 [+7.0, +19.9]** |
| B p=0 | MLP | −45.0 [−59.7, −32.6] | −27.6 |
Valid: 0/80 non-finite. Registered readings: **A hurts, B ties.**
Descriptive, Tier 1: share of dose explained by the current observation (closed-loop) is Pos 0.26, MLP 0.48, LSTM 0.34, ODE 0.23 (median over 40 cells).
### Reading
- The clock arm sits with the recurrent arms, not with the MLP: at Tiers 0/1 it ties LSTM and ODE-RNN and trails the MLP by about as much as they do. Any input beyond the current burden lets PQN fit a time-indexed schedule. Its closed-loop share halves, to the ODE-RNN's. That is a mechanism shared by all three, offered post hoc, untested.
- Under missed visits the clock neither helps nor hurts on the mean. Like the LSTM ([L1136](RESEARCH_LOG.md#L1136)), it beats MTD for the median patient at p = 0.5 and not on the mean.
- So "memory hurts" in the paper is better stated as "any state beyond the current observation hurt at this budget". The ≤ 15-d headroom bound ([L1310](RESEARCH_LOG.md#L1310)) is unaffected.
### Decision
Paper: positional MLP added as an arm; the regular-visit sentence is revised to the narrower claim. No follow-up registered; a test of the shortcut mechanism (e.g. clock-dropout) would be a new registration.

## [2026-10-17] — V8: scrambling the clock of V7's positional MLPs moves U by −14.6 d [−44.2, +10.2] at Tier 1: inconclusive, the schedule reading is neither confirmed nor refuted
**Verdict:** OPEN — the clock matters on average (all four point estimates negative; Tier 0 CI excludes 0) but not by the registered 10-d gate, and per-seed effects span −117 to +32 d · **Gate:** Tier 1 frozen−intact −14.6 [−44.2, +10.2] mean, −14.5 [−57.7, +7.9] median / TTP median 175 [126–259] intact vs 161 [35–343] frozen / exposure median 211.8 vs 178.5 d / burden integral median 9.7e5 vs 1.17e6 / pooled Wilcoxon p=3e-36 (patients pooled across seeds, descriptive only) / 10 seeds × 4 w · **Artifacts:** research/v8_clockscramble.py, research/outputs/v8_clockscramble.json, v8_shards/
### Context
V7 ([L1414](RESEARCH_LOG.md#L1414)) read the positional MLP's 57-d deficit as PQN fitting a time-indexed schedule, post hoc and untested. V8 tests that reading without retraining: the 40 saved Part-A policies are rolled out with the clock input replaced.
### What was tried
Registered as V8 in PREREG.md [2026-10-17], before any cell ran: `frozen` (clock = 178.5 d every visit), `reversed` (357 d − elapsed), `intact`. Primary: frozen − intact on U at Tier 1, seed+patient bootstrap. "Schedule" iff upper CI < −10 d; "ignores" iff CI inside (−10, +10). `intact` had to reproduce V7's shards to 1e-6: it did (all 40 cells). One process per (w, seed): a single process died on the JIT mapping limit ([L207](RESEARCH_LOG.md#L207)); no result was seen before the rerun.
### Results
| Contrast (− intact) | Tier 0 | Tier 1 |
|---|---|---|
| frozen, mean | −21.1 [−45.2, −1.3] | −14.6 [−44.2, +10.2] |
| frozen, median | −0.2 [−32.2, +5.1] | −14.5 [−57.7, +7.9] |
| reversed, mean | −29.7 [−56.3, −7.3] | −22.8 [−54.5, +4.3] |
Per seed (Tier 1, frozen): −3.0, +17.9, −116.7, +31.9, −5.4, −34.9, +14.5, −1.0, +24.1, −73.1. Per w: −5.7, −20.6, +6.0, −38.0.
Valid: 0 non-finite. Frozen doses less (median exposure 178.5 vs 211.8 d), progresses earlier in the lower quartile (TTP q25 35 vs 126 d) and later in the upper (q75 343 vs 259 d). Closed-loop share 0.27 intact, 0.34 frozen, 0.19 reversed (Tier 1).
Registered reading: **inconclusive** (upper CI +10.2 > −10, lower −44.2 < −10). Not a registered pass or fail: **met as registered** (a registered outcome was reached; no amendment).
### Decision
- The paper's "any state beyond the burden lets the trainer fit a schedule" stays a reading, now with one test: removing the clock costs about 15–21 d on average, and the sign flips in 4/10 seeds, so some policies use it and some do not. It does not explain the whole 57-d deficit.
- Exposure falls with a frozen clock and closed-loop share rises, which fits a clock-keyed dosing level more than a clock-keyed switch time; untested.
- No follow-up registered. Clock-dropout training would be a new registration.

## [2026-10-17] — W8: with a known next-visit time the ODE-RNN is BELOW the MLP on held-out schedules (−21.4 d [−42.8, −2.9]), but the lookahead barely helps the MLP, so the test is weak
**Verdict:** OPEN — registered readings P1 deficit, P2 ties, manipulation check ties (+9.7 [−0.4, +19.4]); no continuous-time edge from knowing the schedule, yet the MLP did not use the lookahead either · **Gate:** U mean [seed+patient CI] / ode_L Tier-0 held-out TTP median 343 [238–357] vs mlp_L 357 [266–357] / median exposure 320 vs 192 d / pooled medians in table / per-w Wilcoxon (ODE greater) p 0.93–1.0 / 10 seeds × 4 w · **Artifacts:** research/w8_scheduled.py, research/outputs/w8_scheduled.json, w8_shards/
### Context
CLOSED #2 reserved one re-open: scheduled visits where the next visit's time is known when dosing ([L766](RESEARCH_LOG.md#L766)). Registered as W8 in PREREG.md [2026-10-17] (+ amendment before any cell: lstm_L is −2.0% params, schedule C has 27 visits).
### What was tried
Strobl ranges, Tier 0, PQN as S1. Train on schedule A = gaps (1, 3); test on A and the held-out B = (2, 2), C = (1, 2, 3), no retraining. Arms given slots-to-next-visit: mlp_L (386), lstm_L (25 + Δt), ode_L (62), all 3028–3091 Q-path params; control mlp_blind (S1's MLP). Seeds 61–70, w ∈ {0.05, 0.1, 0.2, 0.4}; 0/160 cells non-finite. Intention-to-treat.
### Results
| Contrast (mean U, d) | mean [CI] | median [CI] |
|---|---|---|
| P1 ode_L − mlp_L, held out (B, C) | −21.4 [−42.8, −2.9] | −15.4 [−24.0, −7.6] |
| P2 ode_L − mlp_L, trained (A) | +3.7 [−13.3, +19.1] | −6.5 [−13.1, +2.1] |
| check mlp_L − mlp_blind (A) | +9.7 [−0.4, +19.4] | +0.0 [−0.1, +2.3] |
| ode_L − lstm_L, held out | +7.2 [−16.5, +29.9] | +7.0 [−8.1, +23.0] |
| ode_L − MTD, A / B / C | −7.3 / −8.7 / −12.0 (all CIs cover 0) | +14.7 / +11.6 / +12.9 |
| mlp_L − MTD, A / B / C | −11.0 / +23.1 [+16.0, +29.2] / −0.9 | +17.2 / +32.4 / +22.0 |
ode_L doses far more (median exposure 320 vs 192 d for mlp_L on B) and is least closed-loop (0.26 vs 0.55).
### Decision
- Registered: P1 deficit and P2 ties ⇒ the prereg kill says CLOSED #2 extends to scheduled visits. But the manipulation check did not exclude 0 and its median is 0.0: the MLP ignored the lookahead, so the run does not show that knowing the schedule is useless to a continuous-time state, only that this ODE-RNN under this PQN budget did not exploit it and generalised worse than the MLP.
- So: no LAW L1 exception, and no new CLOSED item until a variant makes the lookahead usable (not registered; that is a new mechanism). Paper: one sentence in Limitations, replacing "scheduled gaps ... are untested".

## [2026-10-17] — V9: training the positional MLP with its clock hidden half the time recovers the whole deficit (+54.4 d [+44.9, +62.3] over the positional MLP at Tier 1; −2.4 [−5.2, +0.4] vs the MLP)
**Verdict:** OPEN — the registered reading is "shortcut confirmed": V7's 57-d deficit is a clock-keyed schedule that the trainer fits, not the clock's information, and clock-dropout training removes it · **Gate:** Tier 1 cd_show−pos_mlp +54.4 [+44.9, +62.3] mean, +49.4 [+35.6, +61.0] median / TTP median 273 [189–357] vs pos 175 [126–259] / median exposure 260.8 vs 211.8 d / closed-loop share 0.33 vs 0.27 / 40/40 cells finite / 10 seeds × 4 w · **Artifacts:** research/v9_clockdropout.py, research/outputs/v9_clockdropout.json, v9_shards/
### Context
V7 ([L1414](RESEARCH_LOG.md#L1414)): positional MLP −56.8 d vs MLP. V8 ([L1437](RESEARCH_LOG.md#L1437)) scrambling the trained clock cost only −14.6 [−44.2, +10.2], so the deficit was not simply "uses the clock". V9 asks the training-side question: if the trainer is made to cope without the clock half the time, does the deficit go?
### What was tried
Registered as V9 in PREREG.md [2026-10-17] before any cell ran. `cd_mlp` = V7's arm (width 386, Q-path 3091) whose clock input is replaced by its mid-episode value when the first observation is positive (~half the episodes). Seeds 31–40, w ∈ {0.05, 0.1, 0.2, 0.4}, trained at p = 0, Tiers 0/1, paired against S1's and V7's shards. Tested `show` (primary) and `hide`.
### Results
| Contrast (mean U, d) | Tier 0 | Tier 1 |
|---|---|---|
| cd_show − pos_mlp | +45.5 [+33.7, +57.3] | **+54.4 [+44.9, +62.3]** |
| cd_show − mlp | −7.5 [−10.3, −4.5] | −2.4 [−5.2, +0.4] |
| cd_hide − mlp | −5.3 [−7.4, −3.0] | −8.3 [−11.4, −5.1] |
| cd_show − lstm | +34.5 [+17.4, +60.8] | +41.7 [+26.0, +63.8] |
| cd_show − ode_rnn | +42.9 [+28.9, +57.9] | +47.6 [+36.2, +60.2] |
| cd_hide − cd_show | +2.2 [−0.1, +4.7] | −6.0 [−9.0, −2.5] |
Tier-1 per-seed cd_show − pos: +19 to +72, all 10 positive. Closed-loop share 0.33 (show) / 0.41 (hide) vs pos 0.27, MLP 0.46.
### Decision
- Registered reading: shortcut confirmed (lower CI +44.9 > +10). The paper's schedule sentence becomes a tested statement. Caveat: cd_mlp is about as good as the MLP, not better; the clock carries no benefit at regular visits.
- Suggested, not tested: the LSTM and ODE-RNN deficits (−44 to −50 d vs the MLP) may be the same optimisation failure. cd_show beats them by 42–48 d with the same trainer, but a recurrent arm trained with state dropout is a new registration and a new mechanism.
- Cost of the check: V9 changes only the training input; the MLP-vs-recurrent ordering is unchanged until that arm is run.

## [2026-10-17] — W9: with schedules varying across patients, the ODE-RNN beats the MLP on held-out schedules (+34.8 d [+16.1, +53.7]), but the lookahead is again unused by the MLP and the LSTM does as well
**Verdict:** OPEN — the registered gate failed (lookahead not usable, +1.4 [−10.3, +14.4]), so by registration nothing is closed and no further lookahead variant is registered; P1 reads "edge" but is not specific to continuous time (ODE−LSTM −4.6 [−20.3, +9.2]) · **Gate:** held-out (C, D, E) ode_L−mlp_L +34.8 [+16.1, +53.7] mean, +20.2 [+2.4, +46.4] median / TTP median 294 [168–357] ode_L vs 189 [126–315] mlp_L at D / median exposure 191 vs 178 d / closed-loop 0.24 vs 0.50 / Wilcoxon not computed for the primary / 10 seeds × 4 w · **Artifacts:** research/w9_pool.py, research/outputs/w9_pool.json, w9_shards/
### Context
W8 ([L1457](RESEARCH_LOG.md#L1457)) trained on one schedule and the MLP ignored the lookahead. W9 trains on a pool of five schedules (gaps ≤ 3) so the next-visit time varies between patients, and tests on A = (1,3) (in pool) and held-out C = (1,2,3), D = (4,), E = (1,1,4), D and E with gaps longer than any trained.
### What was tried
Registered in PREREG.md [2026-10-17] W9 (+ erratum on the L-link) before any cell ran. Arms mlp_L (386), lstm_L (25 + Δt), ode_L (62), each shown slots-to-next-visit; control mlp_blind (441). Seeds 71–80, w ∈ {0.05, 0.1, 0.2, 0.4}; 0/240 cells non-finite. The gate was decided first: mlp_L − mlp_blind over C, D, E, usable iff lower CI > 0.
### Results
| Contrast (mean U, d) | mean [CI] | median [CI] |
|---|---|---|
| GATE mlp_L − mlp_blind, held out | +1.4 [−10.3, +14.4] | +0.1 [−5.5, +5.3] |
| P1 ode_L − mlp_L, held out | +34.8 [+16.1, +53.7] | +20.2 [+2.4, +46.4] |
| ode_L − mlp_L, D and E only | +45.4 [+23.9, +67.1] | +32.1 [+8.2, +67.5] |
| ode_L − mlp_L, trained schedule A | +23.3 [+7.0, +38.6] | +0.6 [−7.2, +14.7] |
| ode_L − lstm_L, held out | −4.6 [−20.3, +9.2] | +5.0 [−4.3, +14.2] |
Post hoc (not registered): lstm_L − mlp_L +39.4 [+24.4, +53.4]; ode_L − mlp_blind +36.2 [+21.8, +49.8]; ode_L − MTD −7.5 [−19.1, +3.5] (median +21.5); mlp_L − MTD −42.3 [−53.1, −31.5].
### Decision
- Gate failed: the lookahead input is not what any arm exploits, so W9 does not test the known-schedule mechanism. By registration nothing is closed (CLOSED #2 keeps its reserved re-open, now spent twice without a usable input), no LAW L1 exception, and no third lookahead variant.
- Separate finding: under schedules that vary between patients, both recurrent arms beat the MLP by 35–45 d. That is the first setting in the program where memory helps, but the ODE-RNN is not better than the LSTM, so it is a memory effect, not continuous time. L1's scope (regular visits, Tiers 0/1) is untouched.
- Paper: Limitations replace the W8 sentence with both runs and say the gate failed; the recurrent-beats-MLP result is reported as post hoc.

## [2026-10-17] — V10: state-dropout training does not remove the recurrent deficit — LSTM +4.3 d [−13.1, +21.5], ODE-RNN +4.2 [−8.8, +18.9] over the plain arms at Tier 1; both still ~40–46 d below the MLP
**Verdict:** OPEN — registered readings "inconclusive" for both arms (the +10-d gate is inside both CIs), but the effect is a tie and the deficit vs the MLP persists (CIs exclude 0), so V9's shortcut does not carry over to the recurrent arms; with memory unavailable the LSTM equals the MLP · **Gate:** Tier 1 lstm_sd_show−lstm +4.3 [−13.1, +21.5] / ode_sd_show−ode_rnn +4.2 [−8.8, +18.9] / medians +4.3, +1.8 / TTP median 210 [147–294] lstm_sd vs 217 [147–294] lstm; 217 [119–322] ode_sd vs 210 [119–301] / median exposure 224, 203 d / closed-loop 0.36, 0.26 / 80/80 cells finite / 10 seeds × 4 w · **Artifacts:** research/v10_statedropout.py, research/outputs/v10_statedropout.json, v10_shards/
### Context
V9 ([L1477](RESEARCH_LOG.md#L1477)): hiding the positional MLP's clock half the time recovered its whole deficit. V10 gives the LSTM and ODE-RNN the same treatment: in episodes with a positive first observation the recurrent state is reset at every visit.
### What was tried
Registered as V10 in PREREG.md [2026-10-17] before any cell ran, with a declared LAW L1 note (no change of trainer, budget or reward). Part A of S1: seeds 31–40, Tiers 0/1, paired against S1's shards, `show` (memory available) primary, `hide` secondary. No parameters added.
### Results
| Contrast (mean U, d) | Tier 0 | Tier 1 |
|---|---|---|
| lstm_sd[show] − lstm | −2.2 [−21.0, +17.7] | +4.3 [−13.1, +21.5] |
| ode_sd[show] − ode_rnn | +3.5 [−11.6, +20.3] | +4.2 [−8.8, +18.9] |
| lstm_sd[show] − mlp | −44.2 [−59.5, −29.5] | −39.7 [−52.1, −28.7] |
| ode_sd[show] − mlp | −46.9 [−58.4, −36.4] | −45.7 [−54.2, −37.7] |
| lstm_sd[hide] − mlp | −2.4 [−8.8, +2.9] | −0.8 [−7.7, +5.0] |
| ode_sd[hide] − mlp | −17.1 [−26.2, −8.9] | −15.1 [−25.9, −5.1] |
| lstm_sd[hide] − lstm_sd[show] | +41.8 [+27.7, +57.0] | +39.0 [+28.7, +51.2] |
| ode_sd[hide] − ode_sd[show] | +29.8 [+14.6, +45.4] | +30.6 [+15.6, +45.9] |
### Decision
- Registered: neither arm is "explained" (lower CIs −13.1 and −8.8 < +10), so the negative reading holds against the fairer baseline: training that can cope without memory does not remove the recurrent deficit.
- The sharper fact is `hide` − `show`: the trained nets do better when their memory is switched off (+39 and +31 d), and the LSTM then equals the MLP. They use memory in a way that hurts, even though half of training did not have it. The mechanism is not the clock's schedule shortcut; it is untested.
- Paper: one sentence in the Results; the title's "an ODE-RNN" claim stays.

## [2026-10-17] — W10: the PQN trainer lets an LSTM use memory when memory pays — headroom shown on 2 of 3 POPGym tasks (RepeatPrevious +1.38, RepeatFirst +1.04; CountRecall +0.04)
**Verdict:** FROZEN — the registered pass rule is met (2 of 3 tasks), so the cancer-task recurrent deficit is not "PQN cannot use memory"; it is specific to this task and training signal · **Gate:** metric is mean return of 512 greedy episodes (perfect +1, chance −0.5), not TTP; MLP −0.50 / −0.52 / −0.92, LSTM +0.88 / +0.52 / −0.88, 10 seeds per arm · **Artifacts:** research/w10_headroom.py, research/outputs/w10_headroom.json, w10_shards/
### Context
Both outside reviews: PQN may simply be unable to train anything stateful, so the recurrent deficit would be a trainer fact. PREREG W10 [2026-10-17] asks whether the same PQN update lets an LSTM beat an MLP on POPGym memory tasks (popjaxrl, unvendored, no licence).
### What was tried
Three tasks (RepeatPreviousEasy, RepeatFirstEasy, CountRecallEasy), 52 steps each; `mlp` 17 924 and `lstm` 18 052 parameters; `adaptive_therapy.rl.pqn` update, 2000 updates × 64 episodes, reward scale 10, untuned; 10 seeds (101–110). Registered as written; not amended. A first launch lost 24 cells to `No module named flax` (another `uv run` re-synced the environment and dropped the `popgym` extra); they were rerun unchanged from the same seeds with the venv python.
### Results
| Task | MLP | LSTM | Δ lstm − mlp (seed bootstrap) | counted | headroom |
|---|---|---|---|---|---|
| RepeatPreviousEasy | −0.498 (sd 0.005) | +0.876 (sd 0.21) | +1.375 [+1.227, +1.489] | yes | yes |
| RepeatFirstEasy | −0.519 (sd 0.029) | +0.523 (sd 0.58) | +1.042 [+0.666, +1.386] | yes | yes |
| CountRecallEasy | −0.924 (sd 0.003) | −0.883 (sd 0.007) | +0.041 [+0.036, +0.046] | yes | no (< 0.2) |
All three pass the manipulation check (MLP < +0.1). LSTM seeds are bimodal on RepeatFirst (4 of 10 finish below +0.1; range −0.53 to +1.02) and mostly saturate on RepeatPrevious (range +0.45 to +1.00); CountRecall is not learned by either arm (both near the −0.92 floor).
### Decision
- Registered rule: ≥2 of 3 counted tasks show headroom ⇒ PASS. The paper says the trainer can use memory on tasks that need it, and that the LSTM's seed-to-seed instability seen on the cancer task also appears here (RepeatFirst).
- Limits: two easy tasks, one trainer, PQN's published recurrent settings at 64 episodes per update; CountRecall shows the trainer does not reach every memory task. Not evidence about ODE-RNNs, and not about why the cancer task differs (V9/V10 bear on that).
- No other task, width or budget is added after seeing this.

## [2026-10-18] — V11: LSTM capacity sweep — 5× and 22× the parameters do not recover the deficit; both stay −47 to −49 d vs the MLP at Tier 1
**Verdict:** OPEN — registered reading "inconclusive" (upper CIs +18.6 and +26.2 exceed the +10-d gate), but both point estimates are negative and the deficit against the MLP persists, so capacity is not what limits the LSTM · **Gate:** Tier 1 U vs 26-unit LSTM: 64 units −5.1 [−26.8, +18.6], 128 units −3.1 [−25.0, +26.2]; median TTP 210 d both (MLP 273, LSTM-26 217), median exposure 268 / 277 (MLP 245) · 10 seeds × 4 w, 0 non-finite · **Artifacts:** research/v11_capacity.py, research/outputs/v11_capacity.json, v11_shards/, v11_policies/
### Context
Two outside reviews: matching the arms at ~3,100 parameters is thin; a 26-unit LSTM may just be too small. PREREG V11 [2026-10-17].
### What was tried
S1 Part A's design (seeds 31–40, Strobl ranges, PQN, trained at p = 0, w ∈ {0.05, 0.1, 0.2, 0.4}) with the LSTM at 64 units (17 284 parameters) and 128 units (67 332), paired against S1's shards. Run as registered, no amendment.
### Results
| Contrast (mean U, d) | Tier 0 | Tier 1 |
|---|---|---|
| lstm_64 − lstm_26 | −5.1 [−29.0, +21.7] | −5.1 [−26.8, +18.6] |
| lstm_128 − lstm_26 | −1.7 [−27.1, +30.8] | −3.1 [−25.0, +26.2] |
| lstm_128 − lstm_64 | +3.3 [−20.7, +25.6] | +2.0 [−19.2, +22.3] |
| lstm_64 − mlp | −47.1 [−60.1, −35.9] | −49.2 [−62.5, −38.3] |
| lstm_128 − mlp | −43.8 [−62.8, −29.0] | −47.1 [−63.9, −33.3] |
Closed-loop share Tier 1: MLP 0.46, LSTM-26 0.33, LSTM-64 0.30, LSTM-128 0.30.
### Decision
- Registered reading: inconclusive for capacity (CIs are wide, ±22 d), so the matched-capacity claim is not withdrawn. The data do not show a capacity effect: estimates are −5 and −3 d against a +10-d gate, and the 128-unit net, 22× larger, does not beat the 64-unit one.
- Against the MLP, "deficit" for both sizes (upper CIs −38 and −33), so no LSTM size reaches the MLP at this trainer and budget.
- Paper: one sentence and a table row. Not varied (declared): MLP width and ODE-RNN size, so a matched-large MLP is still untested.
