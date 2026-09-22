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

*(nothing yet)*

## CLOSED — do not re-litigate

*(nothing yet)*

## Standing constraints (inherited from the spec, not re-derivable here)

- **Burden-only observation.** The agent never sees subpopulations; this is a
  genuine POMDP by construction → [PROJECT.md §7](../PROJECT.md).
- **Exposure is always reported with TTP.** A policy that wins TTP by dosing
  constantly is aggressive, not robust → [PROJECT.md §10](../PROJECT.md).
- **θ is weakly identifiable from burden.** Q3 is robustness *under*
  non-identifiability, never parameter recovery → [PROJECT.md §5](../PROJECT.md).
