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

- **Simulator sound and the three families are mechanistically distinguishable** — untreated log10 dynamic ranges 0.3 / 2.2 / 2.7, distinct turning-point counts → [L13](RESEARCH_LOG.md#L13)
- **Phase 0 gate PASSED: the three families are separable from burden alone** — median balanced accuracy 0.996 [0.993–1.000], 3 seeds, 0% collisions; Q3 needs no reframe → [L48](RESEARCH_LOG.md#L48)
- **…but the margin is SCALE, not mechanism** — a growth-magnitude-only control scores 0.824, itself above the 0.80 gate; report it beside any separability claim → [L48](RESEARCH_LOG.md#L48)
- **θ is not recoverable from burden, worst in Kuznetsov (6/8 flat)** — LV `r_R` flat at a 49-decade CI, independently reproducing arXiv 2608.18387; assess the MoE on cluster purity, never parameter recovery → [L48](RESEARCH_LOG.md#L48)
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

- **Phase 2 FAILED as registered — but it tested feedback vs a fixed schedule, not discrete vs continuous** — the ODE-RNN arm gave all 200 patients one identical exposure in 4/6 cells (open-loop); every MLP cell was closed-loop and won the paired test (227 and 234 patients earlier progression for ODE, p=3.7e-18, 1.0e-36). Closed: *this ODE-RNN under this PPO budget does not learn a feedback policy on LV*. The architecture question is UNTESTED; a fix is a new registration that diagnoses the collapse first, never a re-grade → [L275](RESEARCH_LOG.md#L275), [L312](RESEARCH_LOG.md#L312)

## Standing constraints (inherited from the spec, not re-derivable here)

- **Burden-only observation.** The agent never sees subpopulations; this is a
  genuine POMDP by construction → [PROJECT.md §7](../PROJECT.md).
- **Exposure is always reported with TTP.** A policy that wins TTP by dosing
  constantly is aggressive, not robust → [PROJECT.md §10](../PROJECT.md).
- **θ is weakly identifiable from burden.** Q3 is robustness *under*
  non-identifiability, never parameter recovery → [PROJECT.md §5](../PROJECT.md).
