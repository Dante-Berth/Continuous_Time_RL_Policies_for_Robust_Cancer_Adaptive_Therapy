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
- **Kuznetsov's apparent dynamics depend on the horizon** — 0 turning points at 365 d, ≥2 and a dormancy excursion to ~1e-7 by 1000 d; the sanity-gate window is a modelling choice → [L13](RESEARCH_LOG.md#L13)

## CLOSED — do not re-litigate

*(nothing yet)*

## Standing constraints (inherited from the spec, not re-derivable here)

- **Burden-only observation.** The agent never sees subpopulations; this is a
  genuine POMDP by construction → [PROJECT.md §7](../PROJECT.md).
- **Exposure is always reported with TTP.** A policy that wins TTP by dosing
  constantly is aggressive, not robust → [PROJECT.md §10](../PROJECT.md).
- **θ is weakly identifiable from burden.** Q3 is robustness *under*
  non-identifiability, never parameter recovery → [PROJECT.md §5](../PROJECT.md).
