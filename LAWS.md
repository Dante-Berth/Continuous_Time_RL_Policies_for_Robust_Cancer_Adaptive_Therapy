# LAWS — what this program has earned the right to assume

**Read this first. It is the cheapest gate in the repo.**

[CLOSED.md](CLOSED.md) lists *which ideas* died. This file states *why they keep
dying* — the generalizations behind the closures. Most proposals can be rejected
here, in ten lines, without opening the kill list or any verdict.

Rules of use:

- A law is **not** a preference or a heuristic. Each one was bought with
  out-of-distribution evidence and cites it. On conflict the
  [RESEARCH_LOG](adaptive_therapy/RESEARCH_LOG.md) wins — a law that
  contradicts a logged result is a bug *here*.
- Laws are **stated once**. Elsewhere, reference them as `LAW L3` rather than
  restating the reasoning. Restatements drift independently; in the sibling repo
  one finding ended up written out in 25 separate files, each slightly different.
- Adding a law requires a **second independent confirmation**. One result is a
  verdict and belongs in
  [VERDICTS.md](adaptive_therapy/VERDICTS.md). A law is a pattern across ≥2
  closures.
- Killing a law takes the same standard as any re-open: new data or a new
  mechanism, never a re-parametrization.

---

## L1 — Continuous time buys no edge on LV at regular visits, in or out of range (Tiers 0/1)

**Promoted 2026-09-25** from two independent closures that share one mechanism,
each against a different baseline:

- **vs a memoryless MLP:** Phase 2c, PQN, the first valid run
  ([L424](adaptive_therapy/RESEARCH_LOG.md#L424), [CLOSED #1](CLOSED.md)). The
  paired direction favoured the MLP in all four Phase-2 runs, under two trainers.
- **vs a parameter-matched LSTM:** W3c ([L565](adaptive_therapy/RESEARCH_LOG.md#L565)).
  ODE − LSTM is −15.7 d on U = min(TTP, 357) − w·exposure (not the reward;
  [L739](adaptive_therapy/RESEARCH_LOG.md#L739)), with a seed+patient CI of
  [−43, +22] ([L596](adaptive_therapy/RESEARCH_LOG.md#L596)).

- **On a second θ regime:** S1 Part A re-ran W3c's design on Strobl 2021's
  ranges ([L766](adaptive_therapy/RESEARCH_LOG.md#L766)): ODE−MLP −50.0 d
  [−63.5, −37.9], ODE−LSTM −5.9 [−27.9, +19.9] at Tier 1. The law is not an
  artefact of the provisional box's strong cost of resistance.

**The law says "no edge", not "worse".** Once seeds are resampled, the ODE-RNN
is not distinguishable from the LSTM ([L596](adaptive_therapy/RESEARCH_LOG.md#L596)).
Its deficit against the MLP is a memory/training effect that the LSTM shares,
not something continuous time adds.

**Mechanism — a hypothesis, untested (amended 2026-09-29).** Dosing may be
near-sufficiently a function of the *current* observation. The MLP learns a
burden-threshold switch ([L461](adaptive_therapy/RESEARCH_LOG.md#L461)) and
only ties the one-line 0.8 × baseline rule
([L565](adaptive_therapy/RESEARCH_LOG.md#L565), [L766](adaptive_therapy/RESEARCH_LOG.md#L766)).
S2 could not test it: a reference that also sees S, R and θ scored 63 d
*below* the burden-only MLP, and earned less training reward even on its own
training patients ([L804](adaptive_therapy/RESEARCH_LOG.md#L804)). Under PQN
at this budget, extra state lowers U whether it is remembered or given, so
"nothing to integrate" and "the trainer cannot integrate it" are not yet told
apart. (L461's "≥ 0.94 of doses explained" is 1 by construction for a
memoryless arm; the finding is the threshold shape.)

**Rejects without further work:** any Tier-0/1 re-run on LV proposing a
different continuous-time cell (LTC, Neural CDE), width or curriculum. Each is a
re-parametrization. **A different trainer or budget** is admissible only with
demonstrated headroom: under it, S2's full-information reference must beat the
burden-only MLP on a seed+patient CI. Without that, it is a re-parametrization
too ([L804](adaptive_therapy/RESEARCH_LOG.md#L804)).

**Does not cover:** irregular visits (Tier 4 — closed separately as
[CLOSED #2](CLOSED.md), a different mechanism), a mechanism
shift (Tier 2), within-episode drift (Tier 3), or other ODE families. Each
breaks the "the current observation is enough" premise differently, and must
be tested, not assumed.

**Killed by:** a valid registered run on LV Tiers 0/1 whose seed+patient CI puts
a continuous-time arm above the best matched discrete arm.
