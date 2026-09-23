# CLOSED — program-level dead ends (do not re-litigate)

Cross-cutting verdicts that will otherwise get re-proposed from a new angle.
Each is closed by evidence, with dates and links. **Before proposing a research
direction, check it against this list.**

Re-opening requires **new data or a new mechanism, not a re-parametrization** —
and the reserved re-open condition, if any, is stated inline on the item.

A module-level closure lives in
[adaptive_therapy/VERDICTS.md](adaptive_therapy/VERDICTS.md); it is promoted
here only when it constrains the whole program. A pattern across ≥2 closures is
promoted further, to [LAWS.md](LAWS.md).

---

## 1. No continuous-time edge on LV under a Tier-1 parameter shift — with either trainer

**Closed 2026-09-23** by Phase 2c, the first valid run
([L424](adaptive_therapy/RESEARCH_LOG.md#L424)): both arms closed-loop, and the
ODE-RNN progresses earlier than a memoryless MLP in 264/283 and 337/359 untied
patient pairs. The paired direction favoured the MLP in all four Phase-2 runs,
under PPO ([L275](adaptive_therapy/RESEARCH_LOG.md#L275),
[L312](adaptive_therapy/RESEARCH_LOG.md#L312)) and PQN. Phase 2's kill clause
fired: the multi-mechanism curriculum (old Phase 3) is not built.

**Do not re-propose:** Tier-1 re-runs with a wider or deeper ODE-RNN, more
updates, a tuned learning rate, a multi-mechanism curriculum, or a third RL
algorithm on the same test. Each is a re-parametrization of a settled question.

**Reserved re-open condition** — the one caveat this closure leaves open: the MLP
dosed ~40% more, so "worse at equal drug" is not established. An
exposure-controlled comparison (the TTP-vs-exposure frontier, ROADMAP W2–W3)
showing the ODE-RNN above the MLP at matched exposure re-opens this item.

**Not covered:** irregular observation (Tier 4) is a different axis with its own
mechanism, and is tested fresh as ROADMAP W4.

---

The scope exclusions the spec declares up front are **not** closures — they were
never opened, and they live in [PROJECT.md §2](PROJECT.md) ("explicit
non-questions"): no clinical-outcome claims, minimal PK only, no real-patient RL
in v1.
