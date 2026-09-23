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
