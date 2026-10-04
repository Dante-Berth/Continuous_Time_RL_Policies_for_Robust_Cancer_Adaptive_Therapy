---
id: arxiv-2609.12264
title: "Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning"
authors: "Kidane BS, Motayed MSH, Wang S"
venue: "arXiv:2609.12264v1"
peer_reviewed: preprint
date_published: 2026
date_reviewed: 2026-10-04
pdf_url: https://arxiv.org/abs/2609.12264
provenance: 1
status: reviewed
relevance_tier: 2

object: policy
model_family: other
observability: full-state
endpoint: survival-proxy
seeds: "1 trained policy per agent; 100-patient cohort (p.3, p.5)"
identifiability_contact: no

route: nothing
law_contact: none
headline: "TD3 vs DQN on a 21-subpopulation chemotherapy ODE with FULL state observation, benchmarked against a PMP open-loop optimum; TD3 higher mean tumour reduction (84.1 vs 82.8%, p=0.0078), DQN tighter dosing (p.5)"
verdict_line: "Full observation, one trained policy per agent, means not medians; no memory or continuous-time arm. Related work only: contrasts with our burden-only POMDP."
---

# Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2609.12264.pdf`. Page numbers
> are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `policy` |
| S2 | Model family | `other` |
| S3 | Observability | `full-state` |
| S4 | Endpoint | `survival-proxy` |
| S5 | Seeds | 1 trained policy per agent; 100-patient cohort (p.3, p.5) |

## What it does

A trait-structured tumour ODE (n = 21 resistance classes, dose u in [0, 3],
10-year horizon) with reward = negative incremental optimal-control cost
(p.2-3). DQN (7 discrete doses) and TD3 trained in MATLAB; PMP shooting gives
the nominal open-loop benchmark (87.7% reduction). Robustness: one policy per
agent on 100 patients with +-10% perturbed growth and drug sensitivity (p.3).

## What transfers

- Both agents approximate the PMP optimum on the nominal patient (Table III, p.5).
- Efficacy-consistency trade-off across the cohort (Table IV, p.5).
- The authors state full observation of all 21 subpopulations is "an
  idealization" and point to POMDP formulations as future work (p.3, p.6).

## What does not

- Full-state observation, single trained policy per agent (no seed variance),
  mean +- CI as the headline, +-10% heterogeneity only.

## Contact with our design

Related work, adaptive-therapy paragraph: RL dosing work mostly assumes the
state is observed; ours is burden-only and asks what memory buys.
