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
- Strobl MAR, Gallaher J, West J, Robertson-Tessi M, Maini PK, Anderson ARA. "Turnover Modulates the Need for a Cost of Resistance in Adaptive Therapy." *Cancer Research* 2021. PMC8455086 / PubMed 33172930.
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
