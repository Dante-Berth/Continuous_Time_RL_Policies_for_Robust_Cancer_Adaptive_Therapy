---
id: arxiv-2006.04439
title: "Liquid Time-constant Networks"
authors: "Hasani R, Lechner M, Amini A, Rus D, Grosu R"
venue: "AAAI 2021, 35(9) (arXiv:2006.04439v4)"
peer_reviewed: yes
date_published: 2021
date_reviewed: 2026-10-04
pdf_url: https://arxiv.org/abs/2006.04439
provenance: 1
status: reviewed
relevance_tier: 4

object: model
model_family: none
observability: n/a
endpoint: none
seeds: "5 runs, mean +- std (Table 3)"
identifiability_contact: no

route: control
law_contact: none
headline: "LTC: CT-RNN with input-dependent time constants; 5-70% better than other RNNs in 4/7 supervised tasks, comparable in 3 (p.6); control is named as a natural, untested application (p.7)"
verdict_line: "All evidence is supervised; control is future work in the paper itself. Our arms omit LTC; concede in Limitations, cite as the variant not tested."
---

# Liquid Time-constant Networks

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2006.04439.pdf`. Page numbers
> are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `model` |
| S2 | Model family | `none` |
| S3 | Observability | `n/a` |
| S4 | Endpoint | `none` |
| S5 | Seeds | 5 runs, mean +- std (Table 3) |

## What it does

Neurons are linear first-order ODEs whose time constant is gated by the input
and state, solved with a fused fixed-step solver; bounded, stable dynamics and
a trajectory-length expressivity bound (pp.1-5). Compared with LSTM, CT-RNN
(= ODE-RNN with damping, p.6), CT-GRU and neural ODE.

## What transfers

- 4/7 time-series tasks improved by 5-70%, 3/7 comparable (p.6).
- Person activity: beats CT-RNN/ODE-RNN by a large margin (p.6).
- Limitations (p.7): vanishing gradients on long dependencies; performance
  "majorly influenced" by the solver (explicit Euler hurts); higher time and
  memory cost than neural ODEs.

## What does not

- No RL experiments; "control of robots" is listed as a natural application
  (p.7), not tested.

## Contact with our design

Related work, continuous-time paragraph; Limitations: an LTC/CfC arm is
untested here. Its solver sensitivity echoes our own fixed-step choice.
