# ROADMAP — continuous-time RL for adaptive therapy

**Created:** 2026-09-22 · **Revised:** 2026-09-23 — Phase 2 failed; the publication
track below is what follows from it.
**Scope:** an honest test of whether a *continuous-time* policy schedules
treatment more robustly under biological distribution shift than a *discrete*
one, inside mechanistic tumour ODEs, observing only total burden. Not a clinical
claim, not a sim-to-real result, virtual patients only.

*Lowest rung of the precedence ladder: [LAWS.md](LAWS.md), [CLOSED.md](CLOSED.md)
and [VERDICTS.md](adaptive_therapy/VERDICTS.md) beat this file on any conflict.
[PROJECT.md](PROJECT.md) is the spec this plan serves.*

---

## Why this program exists

[PROJECT.md §3](PROJECT.md) establishes that two of the three obvious lanes are
occupied. RL-schedules-adaptive-therapy is taken (Gallagher 2024). Recurrence
helps under partial observability is taken (arXiv 2605.02552 — *the LSTM-vs-MLP
axis is theirs*). What is unoccupied is the **three-way intersection**:
continuous-time policy × multi-mechanism curriculum × irregular observation.

Anything broader than that intersection can be deflated by pointing at a row of
the §3 table. The scope discipline is the contribution.

## The method — and why a differential equation rather than a wider net

The thesis is an *inductive bias* claim, not a capacity claim: structuring the
policy as an ODE should help precisely because tumour dynamics are an ODE, and
that structure should survive when the driving mechanism is unknown. A bigger
MLP is the null hypothesis, not a weaker baseline.

This is falsifiable in the strong direction: if continuous-time shows no edge on
a single clean family under a Tier-1 shift, the multi-mechanism curriculum will
not rescue it. That is Phase 2, and it is a stop-or-go gate, not a milestone.

---

## What Phase 2 changed

Phase 2's stop-or-go gate **failed as registered on a valid run** ([L424](adaptive_therapy/RESEARCH_LOG.md#L424)):
under a Tier-1 parameter shift the ODE-RNN progresses earlier than a memoryless
MLP, and the paired direction favoured the MLP in all four Phase-2 runs under two
trainers (PPO [L275](adaptive_therapy/RESEARCH_LOG.md#L275), [L312](adaptive_therapy/RESEARCH_LOG.md#L312); PQN [L424](adaptive_therapy/RESEARCH_LOG.md#L424)). The kill clause fires: the
multi-mechanism curriculum is not built. Program-level record: [CLOSED.md #1](CLOSED.md).

What survives is a narrower, better-posed paper. Reaching one valid comparison
also exposed four ways this kind of evaluation misleads — open-loop collapse,
the progressor-only median, TTP bought with dose, and a mis-built shifted cohort
(traps 4–6 below). Each was caught here with evidence and becomes a section of
the write-up.

**The paper this track produces.** Target: an ICLR 2027 workshop (deadlines
~Jan–Feb 2027 per [PROJECT.md §14](PROJECT.md), unverified). NeurIPS 2026
workshop deadlines (~29 Aug) have passed.

> *Under distribution shift, does an adaptive-therapy policy need memory, or
> continuous time?* Pre-registered answer so far: not under a parameter shift
> (Tier 1). Irregular observation (Tier 4) — where continuous time has a
> mechanical reason to help — decides whether the answer is "never here" or
> "only there". **Both outcomes are the paper.**

Still [PROJECT.md §16](PROJECT.md)'s Q2 + Q4 spine; what changed is that the
Tier-1 result is a negative, and the baseline that won is the simplest one.

---

## Phases

### Phase 0 — identifiability + separability gate (½–1 d) — *passed:* [L48](adaptive_therapy/RESEARCH_LOG.md#L48)

Structural/practical identifiability of each family against the **scalar burden
observable**, plus a trajectory-collision diagnostic.

**Route decided (2026-09-22): profile likelihood in Python, not Julia.**
[PROJECT.md §5](PROJECT.md) names `StructuralIdentifiability.jl`, but no Julia is
installed and the check is too important to defer behind a toolchain. Fisher
information + profile likelihood over θ answers the question Q3 actually poses —
*can θ be recovered in practice from burden?* — and handles Hahnfeldt's
non-polynomial `V^(2/3)` directly, the case that needs substitution under the
structural tool anyway. Julia remains the documented alternative for exact
structural results on the two rational systems (LV, Kuznetsov) if a reviewer
needs that guarantee.

**Gate:** if the families are not separable under burden-only observation, Q3 is
reframed *before* any encoder is written. → identifiability report + separability
plot.

### Phase 1 — simulator (1–1.5 d) — *done:* [L88](adaptive_therapy/RESEARCH_LOG.md#L88); GPU re-check [L166](adaptive_therapy/RESEARCH_LOG.md#L166)

Three families in `diffrax` + minimal PK + reward. LHS sampling over per-family
ranges, untreated sanity gate, rejection log. `vmap`/`jit` batched rollout with
stiffness bucketing. → per-family untreated-trajectory plots.

### Phase 2 — falsifiable core (1–2 d) ← **STOP-OR-GO GATE** — *FAILED:* [L424](adaptive_therapy/RESEARCH_LOG.md#L424)

Runs: void cohort [L242](adaptive_therapy/RESEARCH_LOG.md#L242); PPO, ODE arm open-loop [L275](adaptive_therapy/RESEARCH_LOG.md#L275)/[L312](adaptive_therapy/RESEARCH_LOG.md#L312); PPO fixed,
invalid [L391](adaptive_therapy/RESEARCH_LOG.md#L391); PQN, first valid run, fail [L424](adaptive_therapy/RESEARCH_LOG.md#L424).

One family. Discrete MLP vs. Neural-ODE. One clean Tier-1 shift. Full §10
metrics + paired Wilcoxon, multi-seed.

**If continuous-time shows no edge on the simple clean case, write the negative
result and stop.** → one honest comparison figure.

### ~~Phase 3 — multi-ODE + architecture 2×2 (1–2 d)~~ — killed by Phase 2's kill criterion ([L424](adaptive_therapy/RESEARCH_LOG.md#L424))

### ~~Phase 4 — robustness tiers (1 d)~~ — re-scoped into W3–W5 below

Tiers 0/1/4 continue as W3–W4, Tier 2 as W5; Tier 3 is deferred.

### ~~Phase 5 — biology question (0.5–1 d)~~ — moved to Stage 2 (runway, not merit)

### ~~Phase 6 — write-up (ongoing)~~ — replaced by W6

---

## Publication track — W-phases (2026-09-23)

Every W-phase that produces a comparison is registered in
[PREREG.md](adaptive_therapy/PREREG.md) before its run. All use **PQN** — the only
trainer that produced valid closed-loop arms ([L424](adaptive_therapy/RESEARCH_LOG.md#L424)) — plus the closed-loop
validity gate (≥ 0.10 per cell), the capped TTP median, and the one-sided paired
Wilcoxon.

### W1 — Diagnose the Tier-1 loss (≈ 1 h; a diagnostic, not a verdict) — *done:* [L461](adaptive_therapy/RESEARCH_LOG.md#L461); both papers reviewed 1°

On the 12 saved Phase-2c policies, with no retraining:
- **Tier 0** (in-range test patients). ODE-RNN ≈ MLP in-distribution but worse
  under shift ⇒ its learned internal model *hurts* robustness. Worse in both ⇒ a
  training/capacity gap. This decides the paper's wording, never a criterion.
- **Dose vs burden** for the MLP: is it a threshold rule, the structure of
  clinical adaptive therapy?
- Read **arXiv 2605.02552** and **Gallagher 2024** to 1°. They are the two
  occupied lanes, and the framing contrasts with the first ("recurrence helps
  under partial observability"), which the literature rule forbids until read.

### W2 — Exposure frontier instead of a ±10% match (≈ ½ d build)

The ±10% median-exposure match held in 1 of 8 Phase-2 comparisons: each arm picks
its own dose, so matching was luck. Replace it with a declared sweep of exposure
weights `w`, and compare arms on their **TTP-vs-exposure frontiers** over the
exposure range both reach. The exact rule is fixed in W3's prereg before any
sweep runs. This is also [CLOSED.md #1](CLOSED.md)'s reserved re-open condition.

### W3 — Memory vs continuous time, Tiers 0 and 1 (≈ 1 d; registered) — *INVALID as registered:* [L496](adaptive_therapy/RESEARCH_LOG.md#L496); W3b *INVALID:* [L526](adaptive_therapy/RESEARCH_LOG.md#L526); W3c *VALID, both hypotheses FAIL:* [L565](adaptive_therapy/RESEARCH_LOG.md#L565)

Add the spec's arm (b), **LSTM**, Q-path parameter-matched (~3090) with the same
LayerNorm readout. Grid: {MLP, LSTM, ODE-RNN} × `w`-sweep × 3 seeds, each trained
policy evaluated on Tier 0 and Tier 1. MLP vs LSTM isolates **memory**; **LSTM vs
ODE-RNN isolates continuous time** — the comparison Phase 2 could not make.
An arm that fails the closed-loop gate is reported as a trainability result,
never dropped. Compute ≈ 1 h on CPU (PQN: ~20 s per MLP cell, ~130 s per ODE cell).

### W4 — Irregular observation: the decisive test (≈ 1–2 d; registered) — *INVALID as registered:* [L632](adaptive_therapy/RESEARCH_LOG.md#L632); W4b *FAILED — no edge, kill fires:* [L668](adaptive_therapy/RESEARCH_LOG.md#L668)

*Registered 2026-09-25 in [PREREG.md](adaptive_therapy/PREREG.md) W4, which
supersedes this sketch in two places.* (1) Every arm **trains** with 35% of
visits missed: under regular training LSTM + Δt sees Δt = 7 throughout, which
makes it the strawman warned against below. Train-regular survives as the
secondary zero-shot test on W3c's policies. (2) CIs resample seeds and patients
([L596](adaptive_therapy/RESEARCH_LOG.md#L596)), with 10 seeds.

Build: per-visit gaps in `EpisodeConfig`/`rollout` (the policy signature already
takes `dt`; the TTP rule already works in days). What a missed visit means for
dosing — e.g. the last chosen dose continues until the next observed visit — is
fixed **in the prereg**, before any run.

Design: train on regular weekly visits; test with 20% and 50% of visits dropped
(the spec's Tier 4). Arms: MLP, LSTM, **LSTM + Δt input** — the standard strong
discrete baseline for irregular series; without it a win is against a strawman —
and ODE-RNN.
- ODE-RNN above the best discrete arm on the frontier ⇒ a narrow positive:
  continuous time helps under irregular observation, not under parameter shift.
- Not above ⇒ the negative holds on both axes; written as such.

*From the 1° reading (2026-09-24):* Gallagher 2024 trained policies at one
decision interval and found that evaluating at a longer interval lowers TTP
through "premature progression" (Fig. 3C, p.1934), and that missed visits cost
a single-patient policy ≥ 20% TTP (p.1937). Expect every arm to lose TTP under
dropped visits; the W4 question is which arm loses least, on the frontier.
arXiv 2605.02552's recurrent win hides a reward-coupled state from one fixed
patient; it is no prior that memory helps here, where W1 found a memoryless
burden threshold near-sufficient ([L461](adaptive_therapy/RESEARCH_LOG.md#L461)).

### ~~W5 — Mechanism shift, Tier 2 (optional; only with runway after W4)~~ — killed: its precondition failed on both families ([L1004](adaptive_therapy/RESEARCH_LOG.md#L1004))

Precondition, checked before registering: treated Hahnfeldt must show TTP dynamic
range — Phase 1 found untreated Hahnfeldt TTP resolution-limited (7.0 d, zero
IQR, [L88](adaptive_therapy/RESEARCH_LOG.md#L88)). If it does not, Tier 2 is cut and the limitation stated. It is the
first thing §16's minimum paper can lose, not the last.

### W6 — Write-up (≈ 1 week) — *in progress:* draft in `paper/`; primary regime = Strobl 2021 ranges after S1 ([L703](adaptive_therapy/RESEARCH_LOG.md#L703), [L766](adaptive_therapy/RESEARCH_LOG.md#L766))

4–9 pp. Figures: closed-loop validity across trainers (the open-loop pitfall);
TTP-vs-exposure frontiers for Tiers 0/1/4; the MLP's dose-vs-burden rule. A
table of **every** registered run, void and invalid ones included. A pitfalls
section (traps 4–6), framed as found on this simulator, not as general law.
Limitations: simulation only; LV (+ Hahnfeldt if W5); 3 seeds; PQN settings
taken from the paper's Craftax table, not tuned here.

**Sequencing:** W1 → W2 → W3 → W4 → (W5) → W6, about two weeks of work — well
inside the ICLR 2027 window.

---

## Stage 2 — main-conference follow-up (deferred on runway, not merit)

*2026-09-30:* the observation-schedule question has a first answer within LV.
M1 ([L896](adaptive_therapy/RESEARCH_LOG.md#L896)) maps the missed-visit rate
at which feedback stops beating MTD across cost of resistance and turnover.

[PROJECT.md §17](PROJECT.md): the spine should be a finding about the world, not a
tool comparison. The **observation schedule** is the natural one: W4 builds the
machinery, and Q1 asks whether "less frequent monitoring lengthens TTP"
(Gallagher 2024, a single LV model) holds across mechanisms. Preconditions:
Gallagher read to 1°; the Hahnfeldt/Kuznetsov resolution limits of [L88](adaptive_therapy/RESEARCH_LOG.md#L88)
resolved. Real-data replay (Bruchovsky PSA cohort) belongs here too.

*2026-09-30:* the cross-mechanism half is cut. Treated Hahnfeldt and Kuznetsov
have no TTP dynamic range on provisional θ ([L1004](adaptive_therapy/RESEARCH_LOG.md#L1004)),
so Stage 2 is LV-only unless a family is rebuilt from 1°-read sources.

---

## Deferred — described, not deleted

Kept here so a future session sees these were considered and why they are not
being built yet.

- **Tier-3 within-episode drift** — hardest and most clinically real, and where
  the continuous-time advantage *should* show most.
  **Re-open when:** W4 shows a continuous-time edge somewhere. There is no case
  for testing the hardest tier for an edge absent on the easier ones.
- ~~**Q3 mechanism-inference (MoE cluster purity)**~~ — depended on the
  multi-mechanism curriculum, killed with Phase 3 ([L424](adaptive_therapy/RESEARCH_LOG.md#L424)). Re-opens only with
  Phase 3.
- **Q1 biology sweep** and **real-data replay** — moved to Stage 2.
- **Liquid (LTC) / Neural CDE arms** — "purer" continuous-time policies. A plain
  Neural ODE cannot take new observations: it needs jumps (ODE-RNN), a driving
  path (Neural CDE) or input-dependent dynamics (LTC).
  **Re-open when:** W4 is positive, making the question *which* continuous-time
  model. Building them before that is the architectural pile-up §15 warns
  against.
- **Recurrent-TD3 baseline** (§13, "beating only weak baselines") — the baseline
  that won is the weakest; for a negative result the reviewer's question is
  instead "was the ODE arm trained well?", answered by two trainers plus the
  validity gate. **Re-open when:** W4 is positive.

---

## Kill criteria — pre-registered before each phase, not after

Registered in [adaptive_therapy/PREREG.md](adaptive_therapy/PREREG.md) before the
run, with the **declared search space**. No widening, no re-parametrization.

- **Phase 0: families not separable under burden-only** ⇒ reframe Q3 before
  writing an encoder. *Did not fire:* [L48](adaptive_therapy/RESEARCH_LOG.md#L48).
- **Phase 2: no continuous-time edge on the clean single-family case** ⇒ stop.
  Write the negative result. The multi-ODE curriculum does not rescue an
  architecture that loses on the easy case. *Fired:* [L424](adaptive_therapy/RESEARCH_LOG.md#L424).
- **W3/W4: an arm fails the closed-loop gate** ⇒ that arm gets a trainability
  result and no architecture verdict.
- **W4: ODE-RNN not above the best discrete arm (LSTM + Δt included) on the
  exposure frontier** ⇒ the negative holds on both axes. No LTC/Neural-CDE
  rescue inside this paper.
- **Any result that wins TTP by raising cumulative exposure** ⇒ not a robustness
  result. Report both or report neither ([PROJECT.md §10](PROJECT.md)).
- **Any single-seed or mean-reported comparison** ⇒ not a result yet
  ([PROJECT.md §15](PROJECT.md) desk-reject list).

---

## The traps this program must not fall into

1. **`vmap` over an adaptive-step solver does not give each patient its own step
   count.** Under XLA/SIMD the whole batch takes as many steps as the *stiffest*
   patient, so a few near-singular virtual patients make everyone pay worst-case
   and "runs in 30 s" silently becomes false. Fixed-step for the bulk of vmap'd
   rollouts; bucket by stiffness; `Kvaerno5` only for the stiff bucket. This is
   an architecture decision, not an optimisation to do later
   ([PROJECT.md §12](PROJECT.md)). *For this simulator the decision (fixed-step)
   holds but the mechanism was wrong: Kvaerno5 fails on every patient, adaptive
   Tsit5 on none* ([L88](adaptive_therapy/RESEARCH_LOG.md#L88), [L166](adaptive_therapy/RESEARCH_LOG.md#L166)).

2. **The architecture-vs-curriculum confound.** Giving the ODE policy the rich
   multi-mechanism curriculum while the MLP gets the poor one measures the
   curriculum and reports it as the architecture. The 2×2 is what makes the
   claim mean anything, and it must be run as a 2×2 even when three of the four
   cells are uninteresting ([PROJECT.md §8](PROJECT.md)).

3. **Kuznetsov's Hopf bifurcations break a naive TTP rule.** For some θ the
   untreated tumour oscillates. The sanity gate must **not** reject oscillatory
   patients — they are the interesting ones — but "crosses threshold once =
   progression" is wrong for them. Define the non-monotone TTP rule up front,
   before any patient is generated ([PROJECT.md §6.3](PROJECT.md)).

4. **A policy can be an open-loop schedule and still report a respectable TTP.**
   The ODE-RNN gave every patient one identical dosing history in 4/6 PPO cells,
   unnoticed by every standard endpoint ([L312](adaptive_therapy/RESEARCH_LOG.md#L312), [L391](adaptive_therapy/RESEARCH_LOG.md#L391)). Every comparison
   needs the closed-loop validity gate.

5. **The median over progressors rewards letting more patients progress.** It
   inverted the paired result in three runs ([L275](adaptive_therapy/RESEARCH_LOG.md#L275), [L312](adaptive_therapy/RESEARCH_LOG.md#L312), [L391](adaptive_therapy/RESEARCH_LOG.md#L391)). Use
   min(TTP, horizon) over all patients; administrative censoring makes it exact.

6. **Shifting a parameter the initial state is built from.** Scaling K after
   y0 = n₀·K put half the Tier-1 cohort above capacity; with d_D > 1 LV blew up,
   and NaN scored as "never progressed" ([L242](adaptive_therapy/RESEARCH_LOG.md#L242)). Rebuild y0 from the shifted
   θ, and refuse non-finite trajectories.

---

## Honest prior

The continuous-time-beats-discrete claim is plausible but not likely-by-default.
The nearest occupied lane (arXiv 2605.02552) already found that *recurrence*
helps under partial observability, which means a competent LSTM baseline is
strong, and much of the benefit attributed to "continuous time" in other domains
is really benefit from *memory*. The honest expectation is that the edge, if it
exists, shows up on **Tier 4 (irregular sampling)** — where continuous time has
a mechanical reason to help — and is **small or absent on Tier 1**.

A second risk worth naming now: an ODE-structured policy has an inductive bias
for *smooth* dynamics, and treatment is a sequence of boluses. The bias may be
mismatched to the action space even where it fits the state space.

Phase 2 is designed to find that out in two days rather than six weeks, and
[PROJECT.md §15](PROJECT.md) is explicit that a clean negative is a publishable
workshop result. Run it to settle the mechanism, and accept a null.

*The Tier-1 half of this prior held ([L424](adaptive_therapy/RESEARCH_LOG.md#L424)). The Tier-4 half is W4.*

---

## GPU migration (available 2026-09-23)

Three things change, and one logged finding needs re-testing rather than
assuming it transfers.

1. **Re-test the §12 stiffness finding on GPU.** *Done:* [L166](adaptive_therapy/RESEARCH_LOG.md#L166). [L87](adaptive_therapy/RESEARCH_LOG.md#L88)
   records that an adaptive solver thrashes and that fixed-step is both
   accurate and far faster — but that was measured on CPU under XLA's SIMD
   vectorisation, which is exactly the execution model §12's warning is about.
   GPU vmap has different divergence characteristics. The finding may hold, may
   reverse, or may become irrelevant; it must be **re-measured, not carried
   over**, and the result logged as a separate entry that links back to L88.

2. **Drop the thread cap.** `research/phase2_gate.py` caps OMP/Eigen threads to
   work around CPU-side XLA compilation OOM. On GPU that workaround is
   unnecessary and would throttle host-side work; make it conditional on the
   backend rather than deleting it, since the CPU path must stay reproducible.

3. ~~**GPU makes grid size, not wall-clock, the binding constraint.**~~
   Measured false in [L166](adaptive_therapy/RESEARCH_LOG.md#L166): GPU is 9×
   slower for rollouts and 17–40× slower per PPO update. GPU pays off only if a
   later phase batches far wider than 32 patients per update, and that has to
   be measured then. The caution below still holds for any faster backend: **This is not licence to widen a declared search
   space after seeing a result** — a cheaper run does not make a post-hoc grid
   extension any less of a selection event (§20). Larger cohorts belong in the
   NEXT registration, not in a re-run of the current one.
