# VERDICTS — adaptive_therapy

One line per settled question. Full evidence in [RESEARCH_LOG.md](RESEARCH_LOG.md)
at the linked line. **Read this file at session start; open the log only where a
verdict links.**

Scope: continuous-time vs. discrete RL policies for tumour treatment scheduling,
under burden-only observation. See [ROADMAP.md](../ROADMAP.md) for what is
planned and [CLOSED.md](../CLOSED.md) for what is dead program-wide.

## DEPLOYED / FROZEN

*(nothing — this program has produced no gated result)*

## OPEN

- **Only 9.8% of our LV patients lie inside Strobl 2021's ranges — r_R/r_S median 0.30 vs their 0.5–1.0 (a much stronger cost of resistance)**; every LV verdict holds only on these provisional ranges, and W6 is blocked until the owner picks a re-run on Strobl ranges or a declared synthetic regime → [L703](RESEARCH_LOG.md#L703)
- **Recurrent-arm PQN training is chaotic: a CI that holds trained policies fixed is conditional, not architectural** — per-cell Tier-1 mean U spans 49–273 d across 5 seeds (ODE-RNN), 67–264 (LSTM), 177–295 (MLP); a 2e-7 float perturbation moved one ODE cell 132 → 256 d. Architecture CIs resample seeds and patients from W4 on → [L596](RESEARCH_LOG.md#L596)
- **W3b INVALID again — recurrent Q-networks go open-loop in ~1 of 14 PQN cells (4/54; MLP 0/33)**: one failing cell per recurrent arm on fresh seeds; H_ct overlap 1%. RMST removed the ceiling: MLP Tier-1 RMST 291–318 d vs LSTM 138–283, ODE-RNN 93–265; descriptive LSTM−MLP −45 d [−59, −40]. Next registration must fix per-cell vs per-arm validity first → [L526](RESEARCH_LOG.md#L526)
- **W3 INVALID — no memory or continuous-time verdict**: one open-loop cell each in LSTM (0.098) and ODE-RNN (0.071, constant max dose at w=0), frontier overlap 16–19% < 20%. Descriptively the memoryless MLP holds the 357-d horizon in all 30 cell-tier results and the LSTM is least reliable (5/15 cells collapse to low dosing); the capped median saturates at the horizon, so the next registration needs a longer horizon or restricted-mean TTP → [L496](RESEARCH_LOG.md#L496)
- **W1: the ODE-RNN also loses in-distribution (Tier 0: 10/164, 13/263 untied pairs) — L424 is a training/capacity gap, not a robustness gap**; report Tier 0 beside every Tier-1 result. The MLP learned a burden-threshold switch (dose 0 below ~0.8× baseline, full above; ≥0.94 of doses explained (erratum [L554](RESEARCH_LOG.md#L554)) by the current obs) → [L461](RESEARCH_LOG.md#L461)
- **Simulator sound and the three families are mechanistically distinguishable** — untreated log10 dynamic ranges 0.3 / 2.2 / 2.7, distinct turning-point counts → [L13](RESEARCH_LOG.md#L13)
- **Phase 0 gate PASSED: the three families are separable from burden alone** — median balanced accuracy 0.996 [0.993–1.000], 3 seeds, 0% collisions; Q3 needs no reframe → [L48](RESEARCH_LOG.md#L48)
- **…but the margin is SCALE, not mechanism** — a growth-magnitude-only control scores 0.824, itself above the 0.80 gate; report it beside any separability claim → [L48](RESEARCH_LOG.md#L48)
- **θ is not recoverable from burden, worst in Kuznetsov (6/8 flat)** — LV `r_R` flat at a 49-decade CI, independently reproducing arXiv 2608.18387; assess the MoE on cluster purity, never parameter recovery → [L48](RESEARCH_LOG.md#L48)
- **Phase 2b INVALID: 3/6 ODE-RNN cells open-loop even on the fixed trainer; MLP 6/6 closed-loop** — no architecture verdict; the ODE-RNN is fragile under PPO. Next is a fresh registration with another algorithm (SAC-discrete / recurrent TD3), owner's call; use a censoring-aware TTP estimator → [L391](RESEARCH_LOG.md#L391)
- **The ODE-RNN was collapsed before training: default init saturates it to a fixed schedule within ~10 visits** — entropy 1.07 → 0.00 over one untrained episode; plus a zero-gradient entropy bonus and a stale ODE critic. Fixed for both arms (recurrent PPO, exact entropy, field init ×0.1); fixed-trainer ODE ends closed-loop (0.23, 0.11) → [L353](RESEARCH_LOG.md#L353)
- **Phase 2 run 1 is VOID, not a negative result** — Tier-1 y0 built from unshifted K put 43–50% of test patients above capacity; 14–18% diverged to NaN under full dose and scored as censored; re-run as declared AMENDMENT, run 2 final → [L242](RESEARCH_LOG.md#L242)
- **L128's LLVM crash was a deterministic PPO recompile leak, not a transient** — eager `old_logp` rollout recompiled every update; ~270 JIT mappings each crossed `vm.max_map_count`=65530 near update 240; fixed and regression-tested → [L207](RESEARCH_LOG.md#L207) (supersedes [L128](RESEARCH_LOG.md#L128))
- **PROJECT.md §12's stiffness mitigation is wrong for this simulator** — fixed-step accurate to ~1e-4, on CPU and GPU alike (parity 5.6e-7) → [L88](RESEARCH_LOG.md#L88), [L166](RESEARCH_LOG.md#L166)
- **…and L88's mechanism was wrong: Kvaerno5 fails, not adaptive stepping** — Kvaerno5 solves 0/200 patients, flagged or not; adaptive Tsit5 solves 200/200 in ~600 steps; the fast-transient flag predicts nothing → [L166](RESEARCH_LOG.md#L166)
- **GPU is 9× SLOWER than CPU for cohort rollouts** — n=200 fixed-step 1.10 s vs 0.12 s; small sequential solves are launch-bound → [L166](RESEARCH_LOG.md#L166)
- **TTP rule = running-nadir + 28 d confirmation** — fires at d186 vs baseline's d277 on a responder-relapse; both rules reported so rule-sensitivity stays visible → [L88](RESEARCH_LOG.md#L88)
- **Untreated TTP is resolution-limited on Hahnfeldt (7.0 d) and Kuznetsov (14.0 d)** — zero IQR at weekly sampling; LV 28.1 d [21–47] is the only family with untreated dynamic range, so Phase 2 should use it → [L88](RESEARCH_LOG.md#L88)
- **Kuznetsov's apparent dynamics depend on the horizon** — 0 turning points at 365 d, ≥2 and a dormancy excursion to ~1e-7 by 1000 d; the sanity-gate window is a modelling choice → [L13](RESEARCH_LOG.md#L13)

## CLOSED — do not re-litigate

- **W4b FAILED: no continuous-time edge on LV Tier 4 either — ODE-RNN ties the MLP (−6.1 d [−34.7, +12.9]) and is below MTD (−23.7 [−47.5, −5.8])**, intention-to-treat, seed+patient CI, 10 fresh seeds. It degrades least of the learned arms (U 246 → 229 at p 0 → 0.5 vs MLP 281 → 217), but at 50% missed visits every learned arm is below the fixed MTD schedule (best −29.1 [−55.8, −8.2]). W4 (INVALID: its closed-loop gate excluded near-fixed schedules that scored higher, [L632](RESEARCH_LOG.md#L632)) is superseded → [L668](RESEARCH_LOG.md#L668)
- **W3c VALID: no memory edge and no continuous-time edge on LV Tiers 0/1 — MLP > LSTM ≈ ODE-RNN** on the trained objective U = TTP − w·exposure, CIs resampling seeds + patients: LSTM−MLP −82.9 d [−106.8, −58.7]; ODE−LSTM −15.7 d [−43.0, +21.8], so no edge and no established deficit (L565's patient-only [−20.4, −10.8] held the trained policies fixed; "hurts further" withdrawn). Tier 0 within 10 d, so not a robustness effect. The MLP only ties the one-line 0.8 × baseline rule (246.8 vs 245.9) → [L565](RESEARCH_LOG.md#L565), [L596](RESEARCH_LOG.md#L596)
- **No continuous-time edge on LV Tier-1 with either trainer — Phase 2c FAILED on PQN, the first VALID run** — both arms closed-loop (MLP 0.37–0.54, ODE 0.15–0.31); ODE-RNN progresses earlier in 264/283 and 337/359 untied pairs (p(ODE later)=1.0); capped TTP ties at the horizon (w0.05) or ODE −133 d (w0.20). Caveat: MLP doses ~40% more, so "worse at equal drug" is not established by the registered test; the two ODE cells at MLP-like exposure are the worse ones → [L424](RESEARCH_LOG.md#L424)
- **Phase 2 FAILED as registered — but it tested feedback vs a fixed schedule, not discrete vs continuous** — the ODE-RNN arm gave all 200 patients one identical exposure in 4/6 cells (open-loop); every MLP cell was closed-loop and won the paired test (227 and 234 patients earlier progression for ODE, p=3.7e-18, 1.0e-36). Closed: *this ODE-RNN under this PPO budget does not learn a feedback policy on LV*. The architecture question is UNTESTED; a fix is a new registration that diagnoses the collapse first, never a re-grade → [L275](RESEARCH_LOG.md#L275), [L312](RESEARCH_LOG.md#L312)

## Standing constraints (inherited from the spec, not re-derivable here)

- **Burden-only observation.** The agent never sees subpopulations; this is a
  genuine POMDP by construction → [PROJECT.md §7](../PROJECT.md).
- **Exposure is always reported with TTP.** A policy that wins TTP by dosing
  constantly is aggressive, not robust → [PROJECT.md §10](../PROJECT.md).
- **θ is weakly identifiable from burden.** Q3 is robustness *under*
  non-identifiability, never parameter recovery → [PROJECT.md §5](../PROJECT.md).
