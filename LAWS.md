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

**Mechanism — measured without a trainer (amended 2026-10-02).** "The current
observation is enough" is **wrong as stated**. A per-patient full-information
search beats the MLP by 23–26 d (O1, [L1279](adaptive_therapy/RESEARCH_LOG.md#L1279)).
That splits into thirds (O1b, [L1310](adaptive_therapy/RESEARCH_LOG.md#L1310),
partly post hoc): ~11 d a memoryless threshold near 1.0 × baseline that the
MLP could represent, ~9 d exact observation, ~7 d patient-specific
information. Memory's ceiling is the last two, at most ~15 d. Recurrent
training costs 40–50 d. L1's deficit is a learning cost larger than the
available benefit, not an absence of benefit. History of the question:
dosing was hypothesised to be near-sufficiently a function of the *current*
observation. The MLP learns a
burden-threshold switch ([L461](adaptive_therapy/RESEARCH_LOG.md#L461)) and
only ties the one-line 0.8 × baseline rule
([L565](adaptive_therapy/RESEARCH_LOG.md#L565), [L766](adaptive_therapy/RESEARCH_LOG.md#L766)).
S2 could not test it: a reference that also sees S, R and θ scored 63 d
*below* the burden-only MLP, and earned less training reward even on its own
training patients ([L804](adaptive_therapy/RESEARCH_LOG.md#L804)). Under PQN
at this budget, extra state lowers U whether it is remembered or given, so
"nothing to integrate" and "the trainer cannot integrate it" are not yet told
apart. At PQN's published parallelism (1024 patients per update, S3) the
reference is still 41 d below the MLP
([L840](adaptive_therapy/RESEARCH_LOG.md#L840)). With 10× the gradient steps (T1) it ties the MLP on
training reward yet trails it by 20 d on U
([L1042](adaptive_therapy/RESEARCH_LOG.md#L1042)): the trainer no longer binds,
the reward does (it has no TTP term). With a TTP-bearing reward (V2) the reference reaches the MLP on mean U (−1.4 d [−4.1, +1.2]) and beats it on most patients, but not by the 10 d gate, and earns less training reward ([L1344](adaptive_therapy/RESEARCH_LOG.md#L1344)): the reward was part of the gap, the trainer the rest. The memory arms
(not the reference) earn less of that reward than the MLP
([L1136](adaptive_therapy/RESEARCH_LOG.md#L1136)), so the L1 deficit itself is
optimisation, not misalignment. (L461's "≥ 0.94 of doses explained" is 1 by construction for a
memoryless arm; the finding is the threshold shape.)

**Rejects without further work:** any Tier-0/1 re-run on LV proposing a
different continuous-time cell (LTC, Neural CDE), width or curriculum. Each is a
re-parametrization. **A different trainer, budget or reward** is admissible only with
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
