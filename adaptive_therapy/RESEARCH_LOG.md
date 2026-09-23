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
