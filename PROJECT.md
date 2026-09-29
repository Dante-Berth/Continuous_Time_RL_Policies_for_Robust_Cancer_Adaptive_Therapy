# Continuous-Time RL Policies for Robust Cancer Adaptive Therapy
### Unified project document — roadmap + literature review + workshop plan

**Target venue:** an AI-conference workshop — NeurIPS **DLDE** (*Symbiosis of Deep Learning & Differential Equations*, the natural home; where DN-ODE was published) or an ICLR 2027 *AI-for-Science / ML-for-Health / physics-informed-ML* workshop (deadlines ~Feb 2027).
**Working repo:** `neural-ode-adaptive-therapy`
**Stack (locked):** JAX + `equinox` + `diffrax` + `optax`, custom PPO/A3C loop.
**Discipline rule (non-negotiable):** no citation enters this doc without a resolving DOI/venue. This project already had one AI-generated summary with *fabricated* citations (a phantom "late-2025 UDE tumor paper" with a broken `[cite: topic-1.2.3]` placeholder, plus bare source-tags "AACR Journals / People / OpenReview"). Every reference in §12 was manually verified.

---

## PART I — FRAMING

### 1. What this is (and is not)
**Is:** an honest test of whether a *continuous-time* RL policy (Neural-ODE / Liquid Time-Constant network) schedules cancer treatment more robustly under biological distribution shift than a *discrete* policy (MLP / LSTM), when both are trained inside mechanistic tumor-ODE simulators and observe only total tumor burden.

**Is not:** a claim of clinical validity, a cure for the sim-to-real gap, or (yet) a real-patient study. Virtual patients only in v1.

**One-sentence thesis:** *Does structuring the policy as a differential equation give it an inductive bias for tumor dynamics that survives when the underlying biology is unknown, drifting, or from an unseen mechanism — precisely the regime where the parameters are provably not identifiable from burden data?*

### 2. Research questions (ranked, falsifiable)

**Primary — pick ONE as the paper's spine:**
- **Q1 (mechanism-generalization of a clinical principle):** Gallagher 2024 found that *reducing monitoring/decision frequency increases Time-to-Progression* — from a single Lotka-Volterra model. **Does this hold across mechanistically distinct tumor models (angiogenesis, immune-predation), or is it an artifact of resource competition?** The *negative* answer is the stronger paper: it shows a cited clinical heuristic doesn't generalize.
- **Q1-alt (interpretability survival):** Gallagher's policy collapsed to a single burden threshold. **Does that interpretable structure survive when the driving mechanism is unknown and varying?** If not → an important negative result for the whole "personalized RL for oncology" program.

**Secondary (become the ablations):**
- **Q2 (architecture):** Does a continuous-time policy beat a discrete one under shift, *with RL algorithm, reward, and curriculum held fixed*?
- **Q3 (robustness under non-identifiability):** Given that tumor parameters are *provably weakly identifiable* from burden (see §5), does the agent stay robust *even though it cannot recover them*? — reframed from "can it infer the parameters," which is partly answered *no*.
- **Q4 (irregular sampling):** Does the continuous-time policy degrade more gracefully than discrete baselines when 20–50% of monitoring visits are dropped?

**Explicit non-questions (scope discipline):** no clinical-outcome claims; minimal PK only; no real-patient RL in v1 (real data enters as a calibration/validation touch, §11).

---

## PART II — POSITIONING (what's taken, what's open)

### 3. The competitive landscape
Ordered by closeness. The honest headline: the *"RL schedules adaptive therapy"* and *"recurrence helps under partial observability"* lanes are **occupied**. The open ground is the **three-way intersection**.

| Work | Setup | Overlap | Your differentiator |
|---|---|---|---|
| **Gallagher et al., *Cancer Res* 2024** | A3C DRL on LV prostate; less-frequent decisions ↑ TTP | same baseline & env | you test mechanism-*independence* (Q1) |
| **Recurrent DRL for Chemo under Partial Observability**, arXiv 2605.02552 (2026) | recurrent TD3 (LSTM) vs. feed-forward on AhnChemoEnv/DTR-Bench; recurrence helps under POMDP | **the LSTM-vs-MLP axis is theirs** | continuous-time vs. their LSTM; they *fix* PK/PD, you vary mechanism + do irregular sampling |
| **Adaptive Chemo under Tumor Heterogeneity via RL**, arXiv 2609.12264 (2026) | TD3 vs. DQN, **full observation**, 100-patient generalization | parametric heterogeneity | they assume full subpopulation observation (their stated limitation); you are burden-only POMDP |
| **Adaptive therapy under uncertainty**, arXiv 2608.18387 (2025) | Bayesian calibration (not RL); identifiability + transport failure | the OOD/uncertainty framing | you *act* with a policy; they only forecast |

**Unoccupied intersection = the paper:** continuous-time policy × multi-mechanism curriculum × irregular/missing observations, under a tiered OOD protocol with exposure-controlled metrics. Anything broader can be deflated by pointing at the rows above.

### 4. Paste-ready novelty statement
> Mechanistic oncology models have served as environments for RL adaptive-therapy policies [Gallagher 2024], and recurrent policies are known to help under partial observability [arXiv 2605.02552]. Separately, continuous-time architectures (ODE-RNN [Rubanova 2019], liquid networks [Chahine 2023]) handle irregular sampling and OOD shift in other domains. Bayesian analyses show tumor parameters are only weakly identifiable from burden and do not transport across schedules [arXiv 2608.18387]. We bring these together: we test whether a **continuous-time RL policy**, trained across a **multi-mechanism** virtual-patient curriculum and evaluated under a **tiered distribution-shift protocol with irregular observation times**, yields exposure-controlled schedules that stay robust precisely where identifiability fails.

---

## PART III — METHODS

### 5. The identifiability landmine (read before building any encoder)
**Paper:** "Adaptive therapy under parametric, structural, and measurement uncertainty," arXiv 2608.18387 (2025), on the *same* Strobl LV model.
- Proves the resistant growth rate `r_R` is **weakly identifiable** under adaptive therapy.
- Parameters fit under one schedule frequently **do not transport** to another — via *both* non-identifiability *and* model misspecification.
- Under a time-to-metastasis metric, **TTP and metastasis-risk pull against each other** — a TTP-maximizing schedule can *worsen* metastasis risk.

**Consequences (edges, not footnotes):**
1. Q3 must be about *robustness under* non-identifiability, not parameter recovery (dodges the scoop, asks a stronger question).
2. This paper is the *theoretical reason* discrete policies should fail under shift — cite it instead of asserting.
3. **Report a second endpoint** (metastasis-proxy / burden-integral) alongside TTP, or a reviewer who knows this paper says you optimized the wrong thing.

**Pre-build check (Phase 0):** structural identifiability of each ODE vs. the scalar burden observable, via `StructuralIdentifiability.jl` (LV & Kuznetsov are rational → direct; Hahnfeldt's `V^(2/3)` is non-polynomial → substitution or practical/Bayesian identifiability). If two families share a non-identifiable subspace, the mixture encoder *cannot* separate them — know this on day 1, not after a week of degenerate training runs. **Highest-leverage, least-run check in this whole project.**

### 6. The three mechanistic ODE families
Chosen to be *mechanistically distinct* so a shift means something biological. For each: verified equation form, tunable hyperparameters, and where the real numbers live. **Pull constants from the source tables — do not hardcode from memory.**

**6.1 Lotka-Volterra competition (Gallagher/Strobl baseline)** — sensitive `S`, resistant `R`:
```
dS/dt = r_S·S·(1 − (S+R)/K)·(1 − d_D·u(t)) − d_S·S
dR/dt = r_R·R·(1 − (S+R)/K)               − d_R·R
```
- θ: `r_S, r_R, K, d_S, d_R, d_D`; cost of resistance = `r_R < r_S`.
- **Randomize the four factors Strobl proved matter:** initial resistant fraction `f_R`, proximity to carrying capacity `n_0`, resistant proliferation `r_R`, turnover `d_T`.
- Numbers: Strobl et al. *Cancer Research* 2021 (PMC8455086); realistic *joint* posteriors in arXiv 2608.18387 (Bruchovsky intermittent-ADT cohort) — better than independent uniforms.
- Observable: PSA as noisy, temporally autocorrelated proxy (copy the autocorrelation model from 2608.18387).

**6.2 Hahnfeldt angiogenesis** — volume `V`, dynamic vascular carrying capacity `K`:
```
dV/dt = −a·V·ln(V/K)
dK/dt =  b·V − d·V^(2/3)·K − e·u(t)·K
```
- Signature assumption: inhibition released through tumor surface → `V^(2/3)` (`I/S ~ K^θ x^μ`, `θ+μ = 2/3`).
- θ: `a, b, d, e, K_max`. Numbers: Hahnfeldt et al. 1999; ranges in PLOS One 2015 (journal.pone.0120007); Ledzewicz/Schättler optimal-control sets.
- Bottleneck is *vasculature*, not competition → an LV-tuned threshold has no reason to transfer. This is what makes Tier-2 a real test.

**6.3 Kuznetsov-Taylor immune-predation** — effector `E`, tumor `T` (Gałach 2003 reduction):
```
dE/dt = s + p·E·T/(g+T) − m·E·T − dd·E
dT/dt = a·T·(1 − b·T)          − n·E·T
```
- Produces immunostimulation, "sneaking through," and a **dormant state** — dynamics LV/Hahnfeldt cannot.
- θ: `s, p, g, m, dd, a, b, n`. Numbers: Kuznetsov et al. 1994 (BCL1 lymphoma); non-dim values reused in the delay literature (Wiley 2017, 10.1155/2017/1642976).
- ⚠️ **Hopf bifurcations:** for some θ the untreated tumor *oscillates*. The sanity gate must NOT reject oscillatory patients (they're the interesting ones), but oscillation breaks a naive "crosses threshold once = progression" TTP. **Define the non-monotone TTP rule up front.**

**6.4 Changing hyperparameters in JAX (the whole "curriculum generator"):**
```python
def lotka_volterra(t, y, theta): ...
def hahnfeldt(t, y, theta): ...
def kuznetsov(t, y, theta): ...


# a virtual patient = (model_id, theta); a cohort = arrays thereof
solve_cohort = jax.vmap(solve_one_patient, in_axes=(None, 0, 0))
```
LHS-sample θ inside the per-model ranges → untreated sanity gate → keep survivors. The only real engineering is stiffness bucketing (§9).

### 7. Action / observation / reward model
- **Decision:** discrete (treat/no-treat, matching Gallagher) or discrete dose {0, ½, full}. **Not** continuous infusion — drugs are boluses; continuous control reads as unrealistic to clinical-pharmacology reviewers. Frame: **continuous-time state (Neural ODE), discrete decisions, discrete dose.**
- **Observable:** total tumor burden only (+ noise). Never subpopulations → genuine POMDP.
- **Minimal PK:** single-compartment exponential clearance (dose → effective concentration). Keep it *known* (white-box part).
- **Reward:** +TTP; −penalty for burden > baseline (e.g. +20%); **−cumulative-exposure penalty** (without it the agent cheats by dosing constantly). Report at ≥2 exposure weights.

### 8. Architectures (the ablation)
Swap the **actor architecture inside one RL algorithm**; everything else fixed.

| Fixed | Varies |
|---|---|
| RL algo (A3C/PPO), env, reward | policy net: (a) MLP · (b) LSTM · (c) ODE-RNN/Neural-ODE · (d) Liquid Time-Constant |
| — | curriculum: single-ODE ↔ multi-ODE |

- **Run as a 2×2 (architecture × curriculum)**, not a single "ours wins" line. Confound to avoid: giving the ODE policy the rich curriculum while the MLP gets the poor one.
- **Encoder / irregular-sampling baseline:** don't reinvent it — use **Rubanova ODE-RNN** (hidden state obeys an ODE *between* observations, jumps *at* them). It's your encoder for the MoE latent model *and* the Q4 baseline the pure-Liquid policy must beat.
- **Mechanism-learning model (Q3):** start with **mixture-of-experts latent ODE** (encoder → soft assignment over K learned vector fields), *not* a single universal field. Honest ("a few regimes, figure out which"), diagnosable, and gives a free K=1 vs K=3 ablation.
- **Output-safety layer (both branches):** continuity ≠ safety. It removes high-frequency chatter but does **not** bound dose range or ramp rate. Add explicit sigmoid-bounding + a hard rate limiter, or the "smoothness = safety" claim is unearned.

---

## PART IV — EXPERIMENTAL PROTOCOL

### 9. Robustness tiers (the backbone; "robust" without tiers is unfalsifiable)
- **Tier 0 — in-distribution held-out:** same family, unseen draws. Sanity check, *not* a robustness claim.
- **Tier 1 — parameter extrapolation:** same family, params *outside* training range.
- **Tier 2 — structural/mechanism shift:** a family *never seen in training* (train LV + Kuznetsov, test cold on Hahnfeldt).
- **Tier 3 — within-episode drift:** params change *during* one course (evolving resistance). Hardest, most clinically real, and where the continuous-time advantage should show most (tests online adaptation, not a lucky prior).
- **Tier 4 (separate axis) — irregular/missing observations:** drop 20–50% of visits. Report separately from parameter shift.
- ⚠️ Include **correlated multi-parameter drift** (growth rate + competition coefficient drifting *together*), not just single-parameter ablation — biologically real and a harder, less flattering test.

### 10. Metrics (don't let TTP alone decide)
- **TTP: median + IQR** (not mean — heavy-tailed trajectories).
- **Cumulative drug exposure** — a policy that wins TTP via max dose is aggressive, not robust; this is what makes the comparison to a binary baseline fair.
- **Second endpoint** (metastasis-proxy / burden-integral) — per §5.
- **Paired Wilcoxon signed-rank** across matched virtual patients (not a t-test — its absence is an instant reviewer flag).
- **≥ multiple seeds**, with variance bands — single-seed = desk-reject at a workshop.
- For Q3: **cluster purity of learned mechanism assignment vs. ground-truth family** — the falsifiable version of "it figures out the physics."

---

## PART V — EXECUTION

### 11. Phased plan
- **Phase 0 — Setup + identifiability (½–1 d):** repo/env; §5 structural-identifiability + trajectory-collision diagnostics. **Gate:** if families aren't separable under burden-only, reframe Q3 *before* coding. → identifiability report + separability plot.
- **Phase 1 — Simulator (1–1.5 d):** 3 families in `diffrax` + minimal PK + reward; LHS sampling + untreated sanity gate + rejection log; `vmap`/`jit` batched rollout + stiffness bucketing. → per-family untreated-trajectory plots.
- **Phase 2 — Falsifiable core (1–2 d) ← STOP-OR-GO GATE:** single family, discrete MLP vs. Neural-ODE, one clean Tier-1 shift, full metrics + Wilcoxon. **If continuous-time shows no edge on the simple clean case, the multi-ODE curriculum won't save it → write the negative result and stop.** → one honest comparison figure.
- **Phase 3 — Multi-ODE + architecture 2×2 (1–2 d):** all families; architecture × curriculum; MoE K=1 vs K=3. → 2×2 table + cluster-purity.
- **Phase 4 — Robustness tiers (1 d):** Tiers 0–4 incl. correlated + within-episode drift. → the main robustness figure.
- **Phase 5 — Biology question (0.5–1 d):** Q1 frequency sweep across all 3 families. → frequency-vs-TTP-per-mechanism figure (the "so what for biology").
- **Real-data touch (for the serious version):** replay the learned policy against the Bruchovsky prostate PSA trajectories, or calibrate the virtual cohort to that posterior instead of uniforms. One real-data figure defuses the "pure simulation" discount.
- **Phase 6 — Write-up / visuals (ongoing):** treatment-holiday schedules, trajectories, dose timelines, latent-space plots.

### 12. Stack rationale + the JAX trap
- **`equinox`+`diffrax`+`optax` over Flax-linen:** same author (Kidger); a network *is* a pytree and drops straight into a `diffrax` vector field, differentiable through `grad`/`vmap`. Flax-linen forces `model.apply(params,x)` and threading `params` through solver+adjoint+vmap → a day-3 debugging session inside the adjoint. (If you want Flax, use **NNX**, not linen.)
- **Custom ~150-line PPO/A3C over purejaxrl-unmodified:** purejaxrl assumes Flax-style modules; full-equinox means writing the (short) loop yourself. Decide now, not mid-week.
- ⚠️ **The gotcha nobody warns about:** `vmap` over an *adaptive-step* `diffrax` solver (`PIDController`) does **not** give each patient its own step count — under XLA/SIMD the whole batch takes as many steps as the *stiffest* patient. A few near-singular patients ⇒ everyone pays worst-case; "runs in 30 s" becomes false. Mitigate: fixed-step for the bulk of vmap'd rollouts; bucket by stiffness; `Kvaerno5` only for the stiff bucket.
- **Reusable code:** DTR-Bench/AhnChemoEnv is PyTorch/Gym → won't vmap. Use it as an *external validation anchor* (reproduce one of its numbers in your JAX port = cheap credibility) and as the host for PyTorch baselines. Rubanova ODE-RNN reference code (PyTorch) → port the encoder from it.

### 13. Risk register
| Risk | Source | Mitigation |
|---|---|---|
| Families non-identifiable under burden-only | 2608.18387 proves weak identifiability | §5 day-1 check; reframe/restrict |
| Q3-as-recovery scooped/impossible | 2608.18387 | reframe as robustness *under* non-identifiability |
| LSTM-vs-MLP-under-POMDP is taken | arXiv 2605.02552 | compete on continuous-time × multi-mechanism × irregular |
| TTP-only targets wrong endpoint | 2608.18387 (TTP vs metastasis) | report second endpoint |
| Kuznetsov Hopf oscillations break TTP + gate | Kuznetsov 1994; Wiley 2017 | define non-monotone TTP up front; keep oscillatory patients |
| `vmap`+adaptive solver slowdown | JAX/XLA SIMD | stiffness bucketing / fixed-step bulk |
| NaNs under large shifts | — | lower LR; `Kvaerno5`; clamp θ to plausible ranges |
| No edge on simple case | — | Phase-2 gate → honest negative result |
| Agent wins TTP by over-dosing | — | exposure penalty + always report exposure |
| Confounded architecture-vs-curriculum | — | strict 2×2 |
| Beating only weak baselines | reviewer reflex | implement recurrent-TD3 + MetaPhysiCa; be willing to lose a tier |
| Fabricated citations creep back | prior incident | no citation without resolving DOI/venue |

---

## PART VI — SHAPING THE WORKSHOP PAPER

### 14. Venue + timeline reality
- **ICLR 2026 is over** (workshops were April 26–27, 2026; paper deadline was Jan 31, 2026). "ICLR workshop" now = **ICLR 2027** (workshops announced ~Dec 2026, paper deadlines ~late-Jan/Feb 2027). ~5 months of runway — enough for the real-data + strong-baseline version.
- **Sooner + better-fit:** **NeurIPS DLDE** workshop (your exact topic; DN-ODE's home). NeurIPS workshop deadlines are typically Sept–Oct — the 2026 date may be imminent or just closed; **check it the moment you decide** (unverified here).
- Workshop papers are short (≈4–9 pp), work-in-progress is welcome, often non-archival (so a stronger main-conference version can follow).

### 15. What a workshop reviewer actually rewards / punishes
- **Rewards:** one sharp question; a clean isolating experiment; honesty about limitations; a result that's surprising *or* a clean negative.
- **Punishes (desk-reject signals):** weak-only baselines; single-seed; means-not-medians; over-claimed clinical relevance from simulation; architectural pile-up (MoE+liquid+meta+PINN all at once reads as insecurity).

### 16. Minimum viable workshop paper (what to cut to fit)
Ship **Q2 + Q4** as the spine with **Tiers 1, 2, 4** and the MLP/LSTM/**recurrent-TD3**/ODE-policy comparison, multi-seed, exposure-controlled. That's a complete, honest 4-page story: *continuous-time policies are more robust to mechanism shift and irregular sampling than strong discrete baselines, on a multi-mechanism tumor-ODE benchmark.* Defer to the full paper: the theorem (§17), Q1 biology sweep, Q3 mechanism-inference, real-data replay, Tier-3 within-episode drift.

### 17. If you later want to clear a *main-conference* bar (not workshop)
The four moves that separate "solid specialist paper" from "high-club," ranked by leverage:
1. **Spine = a finding about the world** (Q1 mechanism-(in)dependence), not a tool comparison. The negative answer is the better paper.
2. **Touch real data** (Bruchovsky cohort) — the single biggest credibility jump from toy to grounded.
3. **One formal result** — e.g. *a policy measurable only w.r.t. the identifiable sufficient statistic of the burden signal is minimax-robust to shifts along the non-identifiable manifold.* Turns "we observed robustness" into "we explain why."
4. **Release the benchmark** — the multi-mechanism + tiered-OOD JAX env as a citable "gym for OOD adaptive therapy" (NeurIPS D&B track).
- **The non-technical lever:** a mathematical-oncology co-author (Moffitt IMO — Anderson/Robertson-Tessi; Strobl; Oxford/Maini). Frequently the difference between a workshop and *Cancer Research* / *Nature Machine Intelligence*, and it has nothing to do with your code.

---

## §12-REF. Verified references (all resolve)
- Gallagher K, Strobl MAR, Park DS, et al. *Cancer Research* 2024;84(11):1929–1941.
- Strobl MAR, West J, Viossat Y, Damaghi M, Robertson-Tessi M, Brown JS, Gatenby RA, Maini PK, Anderson ARA. "Turnover Modulates the Need for a Cost of Resistance in Adaptive Therapy." *Cancer Research* 2021;81(4):1135–1147. DOI 10.1158/0008-5472.CAN-20-0806. PMC8455086 / PubMed 33172930. *(Author list corrected 2026-09-29 from PubMed 33172930, owner-approved; the earlier list named Gallaher J, who is not an author.)*
- Hahnfeldt P, Panigrahy D, Folkman J, Hlatky L. "Tumor development under angiogenic signaling." *Cancer Research* 1999. (Ranges: PLOS One 2015, journal.pone.0120007.)
- Kuznetsov VA, Makalkin IA, Taylor MA, Perelson AS. "Nonlinear dynamics of immunogenic tumors." *Bull Math Biol* 1994;56(2):295–321. DOI 10.1007/BF02460644.
- "Adaptive therapy under parametric, structural, and measurement uncertainty." arXiv:2608.18387 (2025).
- "Recurrent Deep Reinforcement Learning for Chemotherapy Control under Partial Observability." arXiv:2605.02552 (2026).
- "Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning." arXiv:2609.12264 (2026).
- Rubanova Y, Chen RTQ, Duvenaud D. "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS 2019, pp. 5320–5330.
- Chahine M, Hasani R, Kao P, Ray A, Shubert R, Lechner M, Amini A, Rus D. "Robust flight navigation out of distribution with liquid neural networks." *Science Robotics* 2023. DOI 10.1126/scirobotics.adc8892.
- Mouli SC, Alam M, Ribeiro B. "MetaPhysiCa: Improving OOD Robustness in Physics-Informed ML." ICLR 2024.
- Xiang J, Qi B, Tang Q, Cerou M, Zhao W. "Data-Driven Neural-ODE Modeling for Breast Cancer Tumor Dynamics and PFS." NeurIPS 2023 Workshop (DLDE).
- Laurie M, Lu J. "Explainable deep learning for tumor dynamic modeling and OS prediction using Neural-ODE (TDNODE)." *npj Systems Biology* 2023. DOI 10.1038/s41540-023-00317-1.
- Zhang J, Cunningham JJ, Brown JS, Gatenby RA. "Integrating evolutionary dynamics into treatment of mCRPC." (AT algorithm; ref'd by Strobl 2021.)


---

## PART VII — HOW THIS PROJECT WORKS (process)

*Engineering method, not research content. Ported from the sibling repos
`quant_crypto` and `herc_llm`. Every rule below exists because they paid for it,
and each one defends a failure mode §13 already names — nothing is imported for
its own sake.*

### 17-bis. Where this comes from, and which shape we take

Two sibling repos run **the same system at two scales**. `quant_crypto` is the
13-module version: a monorepo of independent research lines, one research record
per module, promoting into shared root-level laws. `herc_llm` is the
**single-module** version — one package, one log, one `VERDICTS.md`, the same
ladder and the same skills, flat.

**This project takes the `herc_llm` shape**, because it is one research program
with one question (§2), not thirteen competing lines. Concretely: a single
`adaptive_therapy/RESEARCH_LOG.md`, a single `VERDICTS.md`, and root-level
`LAWS.md` / `CLOSED.md` that start **empty** — this program has produced zero
gated results, so it has earned zero generalizations, and seeding them would be
precisely the dishonesty §19's two-confirmation rule exists to prevent.

Three things are actually being ported:

1. **The precedence ladder** — `RESEARCH_LOG` > `LAWS`/`CLOSED` > spec > roadmap (§18).
2. **The promotion chain** — result → log entry → `VERDICTS` line → `CLOSED`
   item → law, the last step requiring **two independent confirmations** (§19).
3. **`/prereg` and `/logentry`** — the two skills that enforce the chain at the
   only two moments it can break: before a run, and at the end of one (§20).

Everything else here is support for those three.

### 18. The precedence ladder

Documents contradict each other as a program moves. There is **one** ladder, and
it is global — not renegotiated per file. Higher wins, always:

1. `adaptive_therapy/RESEARCH_LOG.md` — a logged, gated result. Nothing overrides it.
2. `LAWS.md` / `CLOSED.md` — generalizations bought with ≥2 confirmations.
3. **This document** — the spec. Read it; do not edit it to match a result.
4. `ROADMAP.md`, `adaptive_therapy/PREREG.md` — committed plans.
5. `CLAUDE.md` and the skills.

A lower-rung doc asserting something about a higher rung is a **staleness bug in
the lower doc** — fix it there, never "resolve" it by editing upward. A law that
contradicts a logged result is a bug in the law.

**Why this matters here:** §13's last row is *"fabricated citations creep back."*
A ladder makes provenance structural rather than habitual — there is exactly one
place a claim can be true, and everything else points at it.

### 19. Past / present / future, and the promotion chain

Three kinds of document, never mixed:

| Kind | File | Rule |
|---|---|---|
| **Past** | `RESEARCH_LOG.md` | Append-only, forever. ≤40 lines/entry, hard cap. Newest at bottom. |
| **Present** | `VERDICTS.md`, `CLOSED.md`, `LAWS.md` | Indices. One line per settled question, linking into the log. |
| **Future** | `ROADMAP.md`, `PREREG.md` | Plans. Lowest rung; they go stale fastest and lose by default. |

Results are promoted, never restated:

```text
run → RESEARCH_LOG entry → VERDICTS line → (cross-cutting) CLOSED item
    → (≥2 closures, same mechanism) LAW
```

A law needs a **second independent confirmation**. One result is a verdict, not
a law. Laws are stated *once* and referenced as `LAW L3` — restatements drift
independently, which is how one finding ended up written out in 25 separate
files in the sibling repo.

**Entry format** (the `**Verdict:**` line is mandatory and machine-greppable):

```text
## [YYYY-MM-DD] — One-line summary
**Verdict:** OPEN|FROZEN|CLOSED — takeaway · **Gate:** … · **Artifacts:** paths
### Context / What was tried / Results / Decision
```

The **Gate** field carries §10's metrics — median TTP + IQR, cumulative
exposure, the second endpoint, Wilcoxon p, seed count. A single-seed or
mean-not-median result is then *visibly non-conforming in the log itself*, which
is the cheapest available defence against §15's desk-reject list.

The 40-line cap is the load-bearing part. In the sibling repo entries ran 1.5–2×
over budget, and that overflow — not the number of entries — is why a log
reached 5,000 lines. **If an entry will not fit, the run needed an artifact, not
more prose.** (This already bit once, on this repo's first entry: the audit
rejected it at 41 lines and the fix was to move a table into the artifact.)

### 20. Pre-registration, and the declared search space

Before any run that could produce a gateable result, append to `PREREG.md`:
hypothesis (mechanism, not vibes), **exact** pass numbers, **kill criteria**,
cohort/plane, planned artifacts, and the **declared search space**. ≤15 lines.

> **If you cannot write a kill criterion, the experiment is not ready to run.**

Step 0 is *check it isn't already dead* against `CLOSED.md`. Any mid-run change
to criteria, grid or cohort is appended as a dated AMENDMENT, never edited in
place — an amendment after seeing partial results is a **selection event** and
must be declared in the log entry.

**Why this matters here:** §11's Phase-2 STOP-OR-GO gate is exactly this, and it
only works if written before the run. Registering it afterwards, once the number
is known, is the difference between a clean negative result (§15's "rewards")
and a rescued one. The declared-search-space clause is what stops "no edge"
becoming "no edge *yet*, let me widen the grid."

### 20-bis. `/logentry` — closing the loop

`/prereg` opens a run; `/logentry` closes it. Run it at the end of any session
that produced a result **or ruled out a direction** — a dead end that never got
logged is one the program will re-propose in three months.

Gather from the session, not from memory, transcribing numbers **from the final
run's artifact** rather than an intermediate diagnostic; if a metric cannot be
traced to a concrete output, say so instead of writing a number. State the
prereg outcome as **met as registered** / **failed as registered** / **criteria
amended mid-run (selection risk)**. Append the entry, never edit an existing
one. **Update `VERDICTS.md` in the same commit** — the log is the archive, the
index is how future sessions find it, and *an entry without an index update is
invisible*. Promote program-level closures to `CLOSED.md` with a re-open
condition. Verify with grep and `make audit`. Commit `[research] …`, staging
only that result's files; do not push.

The pairing is the point: `/prereg` makes the criteria unfalsifiable after the
fact, `/logentry` makes the outcome permanent and findable. Either alone leaks.

### 21. Enforcement — the rules are checks, not discipline

The sibling repo's own verdict, verbatim: *"These rules were honour-system for
months and drifted badly — do not go back to trusting the rule instead of the
check."*

`tools/doc_audit.py` runs in pre-commit and CI:

- **Hard fail (exit 1):** every `RESEARCH_LOG.md#Lnnnn` link resolves.
- **Warn:** append-only violations, index coverage, the 40-line budget,
  past-tense content rotting in forward-looking docs.

Append-only is deliberately *soft*: git history is immutable, so a hard fail
there stays red forever and trains everyone to ignore the tool. Its actual
damage — shifted line links — is caught hard by the link check. That asymmetry
is the design, not an oversight.

**Generated docs.** Any document mixing intent with live state is rendered from
a source of truth with a `--check` mode that fails when stale. Status prose is
*computed, never typed* — this is what stops a file accreting "as of <date>, …"
paragraphs nobody updates.

### 22. Literature with enforced provenance

Directly answers this document's opening discipline rule and §13's last row.

```text
literature/
  REVIEW_PROTOCOL.md   the stable spec — fields, tiers, rules
  PAPERS.md            the index — GENERATED, never hand-edited
  papers/<id>.md       one file per paper, front-matter + prose — source of truth
  pdf/                 gitignored, never committed
```

- **ID scheme:** `doi-…` → `arxiv-…` → `pmc-…`, the filename *and* the table
  key. **Never renumber an ID.**
- **Provenance tier is the whole point:** `1°` = read from the PDF, with a page
  locator on every quantitative cell. `2°` = seeded secondhand. **A `2°` cell
  may not be cited in a design argument** — it may say "this direction exists",
  never "X is true, therefore build Y".
- **The §12-REF list starts entirely at `2°`.** Those citations resolve — that
  was verified by hand — but resolving is not reading. This is the honest
  starting state, and it makes the reading queue generate itself.
- **Status lifecycle:** `queued → screened → reviewed`, or `rejected` with a
  one-line reason **kept forever**. The rejections are what make the index a
  measurement rather than a highlight reel.
- **A paper is not evidence for this program.** Only a logged, gated run on our
  own simulators is. The literature says what to test and what to control for;
  it cannot say what is true here. It sits *below everything* on §18.

This is the mechanism that would have caught the phantom UDE paper and the
broken `[cite: topic-1.2.3]` placeholder automatically, at commit time.

### 23. Killed work stays described

When a phase dies, strike it through, link the entry that killed it, and **keep
the description**:

> *"Left described here rather than deleted, so a future session sees that it
> was considered and why it was not built."*

Applies immediately to §16's deferred set — Tier-3 within-episode drift, Q3
mechanism-inference, the Q1 biology sweep, real-data replay. Each carries a
**re-open condition** inline: re-opening requires **new data or a new mechanism,
not a re-parametrization**.

Two sections every roadmap phase carries, both taken wholesale:

- **The traps this must not fall into** — named failure modes and where each is
  handled in code. Here: the `vmap`-over-adaptive-solver stiffness bomb (§12),
  the architecture-vs-curriculum confound (§8), and Kuznetsov's Hopf
  oscillations breaking a naive TTP rule (§6.3).
- **Honest prior** — the pre-registered expectation, written before the run,
  including the expectation of failure. Here: a competent LSTM baseline is
  strong, much of what is credited to "continuous time" elsewhere is really
  benefit from *memory*, and the edge — if it exists — is likelier on Tier 4
  (irregular sampling, where continuous time has a mechanical reason to help)
  than on Tier 1. Saying so *in advance* is what makes a negative result
  publishable rather than embarrassing.

### 24. Code conventions

- **One home per piece of logic.** A second copy of a TTP calculation or a
  solver is a bug, not a convenience. Import it.
- **`adaptive_therapy/` is imported and tested; `research/` is one-shot
  scripts.** All I/O lives in `research/`, never in the package. The JSON
  artifact is the record; the printed summary is convenience; **log entries
  transcribe from the artifact**.
- **Dated, evidenced comments on every config exception** — *why*, with the
  incident. The single most transferable habit in both sibling repos.
- **Tests assert invariants, not functions**, and are named for the invariant.
- **Explicit seeds threaded as parameters**, never global seeding.
- **Commits:** `[research]`, `[feat]`, `[fix]`, `[chore]`, `[docs]`. Subject
  states the *finding*, not the file list. Stage only the result's own files.

### 25. What this process costs, honestly

It is overhead, and on a 4–9 page workshop paper with ~5 months of runway it is
worth saying what it buys and what it does not.

**Buys:** §13 risks 1, 2, 8 and 12 become structural rather than
vigilance-based; a clean negative stays publishable because it was
pre-registered; and the §12-REF list cannot silently regrow a fabricated entry.

**Does not buy:** anything about whether the science works. The sibling repo ran
this system for months and still closed most of its lines of inquiry. That is
the system functioning correctly — it exists to make dead ends *cheap and
legible*, not to prevent them.

**Skip if:** a mechanism does not defend a named risk. Five root documents is
already near the limit where a newcomer can orient; the sibling repo has
thirteen and acknowledges this as a navigation problem. Do not add a sixth
without a §13 row pointing at it.
