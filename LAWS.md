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
  ODE − LSTM is −15.7 d on the trained objective, with a seed+patient CI of
  [−43, +22] ([L596](adaptive_therapy/RESEARCH_LOG.md#L596)).

**The law says "no edge", not "worse".** Once seeds are resampled, the ODE-RNN
is not distinguishable from the LSTM ([L596](adaptive_therapy/RESEARCH_LOG.md#L596)).
Its deficit against the MLP is a memory/training effect that the LSTM shares,
not something continuous time adds.

**Mechanism.** On LV under burden-only observation, dosing is near-sufficiently
a function of the *current* observation: the MLP learned a burden-threshold
switch that explains ≥ 0.94 of its doses
([L461](adaptive_therapy/RESEARCH_LOG.md#L461)), and it only ties the one-line
0.8 × baseline rule ([L565](adaptive_therapy/RESEARCH_LOG.md#L565)). With
regular visits, a policy has nothing for a continuous-time state to integrate
that the next observation does not already report.

**Rejects without further work:** any Tier-0/1 re-run on LV proposing a
different continuous-time cell (LTC, Neural CDE), width, trainer or curriculum.
Each is a re-parametrization of the mechanism above.

**Does not cover:** irregular visits (Tier 4, registered as W4), a mechanism
shift (Tier 2), within-episode drift (Tier 3), or other ODE families. Each
breaks the "the current observation is enough" premise differently, and must
be tested, not assumed.

**Killed by:** a valid registered run on LV Tiers 0/1 whose seed+patient CI puts
a continuous-time arm above the best matched discrete arm.
