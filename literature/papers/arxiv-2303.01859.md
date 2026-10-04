---
id: arxiv-2303.01859
title: "POPGym: Benchmarking Partially Observable Reinforcement Learning"
authors: "Morad S, Kortvelesy R, Bettini M, Liwicki S, Prorok A"
venue: "ICLR 2023 (arXiv:2303.01859v1)"
peer_reviewed: yes
date_published: 2023
date_reviewed: 2026-10-04
pdf_url: https://arxiv.org/abs/2303.01859
provenance: 1
status: reviewed
relevance_tier: 3

object: dataset
model_family: none
observability: n/a
endpoint: none
seeds: "3 trials x 3 difficulties per env/model, >1700 trials (p.8)"
identifiability_contact: no

route: control
law_contact: none
headline: "13 memory models x 15 POMDPs under PPO: classical RNNs beat modern memory; the memoryless MLP scores highest on almost all navigation tasks, and in many games no memory model beats it (pp.8-9)"
verdict_line: "Precedent for 'MLP beats memory' in RL at scale; also names the positional-MLP baseline we lack (p.9). Cite in related work; concede PosMLP in Limitations."
---

# POPGym: Benchmarking Partially Observable Reinforcement Learning

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2303.01859.pdf`. Page numbers
> are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `dataset` |
| S2 | Model family | `none` |
| S3 | Observability | `n/a` |
| S4 | Endpoint | `none` |
| S5 | Seeds | 3 trials x 3 difficulties per env/model, >1700 trials (p.8) |

## What it does

A library of 15 partially observable environments and 13 memory baselines
(MLP, positional MLP, GRU, LSTM, Elman, IndRNN, LMU, S4D, TCN, frame stack,
FWP, FART, DNC) on RLlib, all trained with PPO and conservative
hyperparameters; >1700 trials (p.8).

## What transfers

- **Supervised learning is a bad proxy for RL** (p.8): models that beat
  LSTM/GRU in SL were among the worst in RL; S4D was unstable.
- **The MLP wins navigation** (p.9): "the MLP received the highest score on
  almost all navigation tasks, beating all memory models".
- **Memory is minimally effective in many games** (p.9): all 13 models converge
  to nearly the same reward, which the authors suspect is PPO's limit.
- **Positional MLP** (MLP + time step) "should be a standard baseline" (p.9).

## What does not

- No neural-ODE / ODE-RNN / LTC arm, no irregular observation times.
- One algorithm (PPO); max-mean episodic reward, a different statistic from
  ours.

## Contact with our design

Related work, memory paragraph: our memoryless MLP winning is consistent with
a broad benchmark finding. The missing positional-MLP arm is a limitation to
state (our policies see no clock).
