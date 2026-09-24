---
id: arxiv-2605.02552
title: "Recurrent Deep Reinforcement Learning for Chemotherapy Control under Partial Observability"
authors: "Kiram FME, Youkana I, Saouli R, Susto GA, Kahloul L"
venue: "arXiv:2605.02552v1, 4 May 2026"
peer_reviewed: preprint
date_published: 2026
date_reviewed: 2026-09-24
pdf_url: https://arxiv.org/abs/2605.02552
provenance: 1
status: reviewed
relevance_tier: 1

object: policy
model_family: other
observability: full-state
endpoint: survival-proxy
seeds: "10, mean ± std (p.4, Table I p.5)"
identifiability_contact: no

route: control
law_contact: none
headline: "Recurrent TD3 (LSTM) 380.4 ± 12.2 vs memoryless TD3 −49.8 ± 252.2 when the NORMAL-cell state is hidden, on ONE fixed patient (Table I, p.5); tumour burden itself is observed (p.2)"
verdict_line: "Occupies memory-helps-under-POMDP, but on a single fixed patient with a hidden toxicity state, means over 10 seeds, and no distribution shift. Our W1 memoryless MLP beat the recurrent arm; its result does not predict ours. It sets the minimum baseline: an LSTM arm must exist (W3)."
---

# Recurrent Deep Reinforcement Learning for Chemotherapy Control under Partial Observability

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2605.02552.pdf`
> (v1, 6 pp.). Page numbers are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `policy` — recurrent TD3 vs TD3 / SAC / rPPO |
| S2 | Model family | `other` — AhnChemoEnv (DTR-Bench), 4-state N/T/I/B ODE (Eq. 4, p.3) |
| S3 | Observability | hides **normal cells** N only; T, I, B observed with multiplicative noise (Eq. 1, p.2). Not burden-only, so `full-state` is the closer screen value. |
| S4 | Endpoint | benchmark return (Eq. 3, p.2); no TTP |
| S5 | Seeds | 10, **mean ± std** (p.4; Table I, p.5) |

## What it does

Recurrent TD3 with separate LSTM actor and critic encoders plus a
current-observation shortcut (after Ni et al. 2022), context length 32 steps,
zero hidden-state init per subsequence (p.3). Compared with TD3 and SAC, under
full and partial observability, and rPPO under partial observability only;
4 × 10⁵ environment steps (p.4). **PK/PD parameters are fixed**, so there is one
patient and no inter-patient variability, by design (abstract, p.1; p.4).

## What transfers

- Under full observability recurrence adds little: rTD3 388.8 ± 4.6 vs TD3
  367.7 ± 7.2 (Table I, p.5).
- Under partial observability the memoryless arms become **unstable**, not
  just worse: TD3 −49.8 ± 252.2, SAC −154.1 ± 219.7, vs rTD3 380.4 ± 12.2 and
  rPPO 308.4 ± 37.0 (Table I, p.5); oscillating evaluation curves (Fig. 3, p.4).
- The authors' mechanism: the hidden N couples into the reward and the safety
  termination (70% of N₀, p.3), so the current observation cannot tell the
  agent how close it is to the constraint (p.5).

## What does not

- **One patient, no shift.** Nothing is tested out of distribution. Our
  question is robustness across a cohort under a θ shift.
- **Hidden variable ≠ ours.** They hide a toxicity state that drives the
  reward and the terminal penalty. Our POMDP hides the S/R split, and W1 shows a
  memoryless threshold on burden already captures ≥ 0.96 of our MLP's doses
  ([L461](../../adaptive_therapy/RESEARCH_LOG.md#L461)).
- Means ± std, no paired test, no exposure reported beside outcome.
- Recurrent arms get an off-policy replay design (32-step subsequences); the
  memoryless arms are stock TD3/SAC. Stability of training, not only
  information, differs between arms.

## Contact with our design

PROJECT.md §3: it occupies the recurrence-vs-MLP lane, so the paper must include
an LSTM arm (ROADMAP W3) and cannot claim "memory helps" as new. Their
recurrent win is **not** a prior for our setting: their hidden variable
enters the reward directly, ours only through slow competition, where a burden
threshold is near-sufficient (Gallagher 2024, doi-10.1158-0008-5472.CAN-23-2040,
p.1935). Contrast, do not cite as agreement.
