---
id: arxiv-2110.05038
title: "Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs"
authors: "Ni T, Eysenbach B, Salakhutdinov R"
venue: "ICML 2022, PMLR 162 (arXiv:2110.05038v3)"
peer_reviewed: yes
date_published: 2022
date_reviewed: 2026-10-04
pdf_url: https://arxiv.org/abs/2110.05038
provenance: 1
status: reviewed
relevance_tier: 3

object: policy
model_family: none
observability: n/a
endpoint: none
seeds: "varies by benchmark (see source figures)"
identifiability_contact: no

route: control
law_contact: L1 contact (oracle below recurrent, p.8)
headline: "Recurrent model-free RL, carefully implemented, matches or beats specialised POMDP methods on 18/21 environments (p.1); oracle policies given the hidden state often UNDERPERFORM it under shared hyperparameters (p.8)"
verdict_line: "Our S2 (full-information reference below the burden-only MLP) has a published precedent: a shared-hyperparameter oracle can lose. Cite as context for S2, and as the 'implementation matters' objection we concede."
---

# Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2110.05038.pdf`. Page numbers
> are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `policy` |
| S2 | Model family | `none` |
| S3 | Observability | `n/a` |
| S4 | Endpoint | `none` |
| S5 | Seeds | varies by benchmark (see source figures) |

## What it does

Revisits the claim that recurrent model-free RL is weak on POMDPs. With design
choices fixed by ablation (separate actor/critic RNNs, an off-policy backbone
such as TD3/SAC, a tuned context length), one recurrent implementation matches
or beats six specialised methods (meta-RL, robust RL, generalisation, credit
assignment) on 18/21 environments (abstract, p.1).

## What transfers

- **Oracles can lose under shared hyperparameters** (p.8): "the oracle policies
  often underperform our implementation … caused by using the same
  hyperparameters for the oracle policies". Same pattern as our S2/S3/T1.
- **Implementation matters** (findings 1–4, p.9): separate actor/critic RNNs;
  off-policy backbone; task-specific context length; inputs must carry enough
  information to infer the hidden state.
- Robust RL (p.7): recurrent average-case training beat a worst-case method on
  worst-case return.

## What does not

- Locomotion and gridworld benchmarks with hidden states fixed per episode or
  occlusion; no irregular observation times, no continuous-time policy.
- Its recipe is tuned per task; ours is deliberately untuned (one PQN setting
  for all arms), so it supports the 'undertuned' objection against us.

## Contact with our design

Related work, memory paragraph. Supports two sentences: memory is a strong
baseline when implemented well (so our negative is about the available benefit,
not about recurrence in general), and a hidden-state oracle under shared
hyperparameters can lose (precedent for S2, [L804](../../adaptive_therapy/RESEARCH_LOG.md#L804)).
