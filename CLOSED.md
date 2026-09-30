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

**Confirmed 2026-09-29 on Strobl 2021's ranges** (S1 Part A,
[L766](adaptive_therapy/RESEARCH_LOG.md#L766)): ODE−MLP −50.0 d [−63.5, −37.9] at
Tier 1 on a seed+patient CI, intention-to-treat, 10 fresh seeds.

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

## 2. No continuous-time edge on LV under missed visits (Tier 4)

**Closed 2026-09-29** by W4b ([L668](adaptive_therapy/RESEARCH_LOG.md#L668)),
registered intention-to-treat after W4's gate proved unfit for Tier 4
([L632](adaptive_therapy/RESEARCH_LOG.md#L632)). Trained and tested with missed
visits, the ODE-RNN ties the MLP (−6.1 d [−34.7, +12.9]) and is below a fixed
MTD schedule (−23.7 [−47.5, −5.8]) on a seed+patient CI. With [LAW L1](LAWS.md)
the negative covers LV Tiers 0, 1 and 4.

**Confirmed 2026-09-29 on Strobl 2021's ranges** (S1 Part B,
[L766](adaptive_therapy/RESEARCH_LOG.md#L766)): ODE−MLP −5.4 [−24.3, +9.3],
ODE−MTD −13.5 [−33.0, +2.3]. No learned arm beats MTD at 50% missed visits in
either regime over the 357-d horizon. Over three years, sparing threshold rules
overtake MTD at high turnover ([L932](adaptive_therapy/RESEARCH_LOG.md#L932)), for
the median patient only where r̂_R ≥ 0.75 and only through drug saved
([L968](adaptive_therapy/RESEARCH_LOG.md#L968)); the
continuous-time closure itself is unaffected.

**Do not re-propose:** LTC, Neural CDE or another continuous-time cell on the
same test; more seeds, other drop rates, a different hold rule, or a different
p_train. Each is a re-parametrization.

**Reserved re-open condition:** a Tier-4 setting where the next visit's timing
is *known* to the policy when it doses (scheduled rather than i.i.d. gaps). W4's
declared risk (1) — i.i.d. gaps make Δt describe only the past — is the one
mechanism this closure did not test.

**Not covered:** mechanism shift (Tier 2), within-episode drift (Tier 3), other
ODE families.

---

## 3. No dosing comparison on Hahnfeldt or Kuznetsov as simulated (Tier 2, cross-mechanism monitoring)

**Closed 2026-09-30** by W5-pre ([L1004](adaptive_therapy/RESEARCH_LOG.md#L1004)),
a pre-registered precondition check with rules only. Treated Hahnfeldt progresses at
the first visit under every policy, MTD included: the drug term cannot outpace
vascular growth. Treated Kuznetsov is all-or-nothing: MTD cures every patient
and every sparing rule progresses at 28–42 d. With no resistant compartment,
MTD is optimal by construction. LV is the program's only family with a
resistance mechanism, so every adaptive-therapy result is LV-only.

**Do not re-propose:** widening or shifting the provisional θ ranges, rescaling
the PK/effect map, another TTP rule, visit interval or policy set on these two
families. Each is a re-parametrization chosen after seeing this result.

**Re-open condition:** Hahnfeldt 1999 / Kuznetsov 1994 read 1°, with θ ranges and
a drug-effect scale taken from them (new data); for Kuznetsov, additionally a
resistance mechanism (a new model).

**Not covered:** other mechanistic families; LV's own Tier-3 drift.

---

The scope exclusions the spec declares up front are **not** closures — they were
never opened, and they live in [PROJECT.md §2](PROJECT.md) ("explicit
non-questions"): no clinical-outcome claims, minimal PK only, no real-patient RL
in v1.
