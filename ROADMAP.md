# ROADMAP — continuous-time RL for adaptive therapy

**Created:** 2026-09-22
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

## Phases

### Phase 0 — identifiability + separability gate (½–1 d)

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

### Phase 1 — simulator (1–1.5 d)

Three families in `diffrax` + minimal PK + reward. LHS sampling over per-family
ranges, untreated sanity gate, rejection log. `vmap`/`jit` batched rollout with
stiffness bucketing. → per-family untreated-trajectory plots.

### Phase 2 — falsifiable core (1–2 d) ← **STOP-OR-GO GATE**

One family. Discrete MLP vs. Neural-ODE. One clean Tier-1 shift. Full §10
metrics + paired Wilcoxon, multi-seed.

**If continuous-time shows no edge on the simple clean case, write the negative
result and stop.** → one honest comparison figure.

### Phase 3 — multi-ODE + architecture 2×2 (1–2 d)

All three families; architecture × curriculum as a strict 2×2; MoE K=1 vs K=3.
→ 2×2 table + cluster purity.

### Phase 4 — robustness tiers (1 d)

Tiers 0–4 including correlated multi-parameter drift and within-episode drift.
→ the main robustness figure.

### Phase 5 — biology question (0.5–1 d)

Q1 frequency sweep across all three families. → frequency-vs-TTP-per-mechanism.

### Phase 6 — write-up (ongoing)

Treatment-holiday schedules, trajectories, dose timelines, latent-space plots.

---

## Deferred — described, not deleted

Kept here so a future session sees these were considered and why they are not
being built yet. Per [PROJECT.md §16](PROJECT.md), the minimum viable workshop
paper is Q2 + Q4 with Tiers 1/2/4.

- **Tier-3 within-episode drift** — hardest and most clinically real, and where
  the continuous-time advantage *should* show most. Deferred because Phases 2–4
  must establish there is an advantage at all first.
  **Re-open when:** Phase 4 passes with an effect large enough to survive a
  harder tier.
- **Q3 mechanism-inference (MoE cluster purity)** — deferred because
  [PROJECT.md §5](PROJECT.md) shows θ is weakly identifiable, so the falsifiable
  version is cluster purity, not parameter recovery.
  **Re-open when:** Phase 0's separability plot shows the families *are*
  separable under burden-only.
- **Q1 biology sweep** — the "so what for biology" figure and the better
  main-conference spine. Deferred on runway, not on merit.
- **Real-data replay (Bruchovsky PSA cohort)** — the single biggest credibility
  jump from toy to grounded. Deferred to the full paper.

---

## Kill criteria — pre-registered before each phase, not after

Registered in [adaptive_therapy/PREREG.md](adaptive_therapy/PREREG.md) before the
run, with the **declared search space**. No widening, no re-parametrization.

- **Phase 0: families not separable under burden-only** ⇒ reframe Q3 before
  writing an encoder. Do not proceed to a mixture model that provably cannot
  separate its components.
- **Phase 2: no continuous-time edge on the clean single-family case** ⇒ stop.
  Write the negative result. The multi-ODE curriculum does not rescue an
  architecture that loses on the easy case.
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
   ([PROJECT.md §12](PROJECT.md)).

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

---

## GPU migration (available 2026-09-23)

Three things change, and one logged finding needs re-testing rather than
assuming it transfers.

1. **Re-test the §12 stiffness finding on GPU.** [L87](adaptive_therapy/RESEARCH_LOG.md#L88)
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

3. **Cohort and grid sizes become the binding constraint, not wall-clock.**
   The Phase 2 grid is 12 runs ≈ 2.2 h on CPU. On GPU this is minutes, which
   makes Phase 3's architecture × curriculum 2×2 and Phase 4's five tiers
   affordable at larger n. **This is not licence to widen a declared search
   space after seeing a result** — a cheaper run does not make a post-hoc grid
   extension any less of a selection event (§20). Larger cohorts belong in the
   NEXT registration, not in a re-run of the current one.
