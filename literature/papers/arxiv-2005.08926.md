---
id: arxiv-2005.08926
title: "Neural Controlled Differential Equations for Irregular Time Series"
authors: "Kidger P, Morrill J, Foster J, Lyons T"
venue: "NeurIPS 2020 (arXiv:2005.08926v2)"
peer_reviewed: yes
date_published: 2020
date_reviewed: 2026-10-04
pdf_url: https://arxiv.org/abs/2005.08926
provenance: 1
status: reviewed
relevance_tier: 4

object: model
model_family: none
observability: n/a
endpoint: none
seeds: "5 repeats per experiment (see tables)"
identifiability_contact: no

route: control
law_contact: none
headline: "Neural CDE beats GRU-dt, GRU-D, GRU-ODE and ODE-RNN on irregular classification (p.5); its gain is dramatic WITH observational intensity and every model degrades without it (p.6)"
verdict_line: "Continuous-time models gain most when the observation pattern is informative; our i.i.d. missed visits carry none. Cite to motivate scheduled/informative gaps as the untested setting."
---

# Neural Controlled Differential Equations for Irregular Time Series

> **Provenance: `1°`.** Read from `literature/pdf/arxiv-2005.08926.pdf`. Page numbers
> are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `model` |
| S2 | Model family | `none` |
| S3 | Observability | `n/a` |
| S4 | Endpoint | `none` |
| S5 | Seeds | 5 repeats per experiment (see tables) |

## What it does

Replaces the ODE-RNN's 'ODE between observations, jump at observations' with a
controlled differential equation driven by an interpolated path of the data,
so observations enter continuously; trainable by adjoints across observations
(pp.1-2). Supervised classification on CharacterTrajectories, PhysioNet sepsis
and Speech Commands.

## What transfers

- With data randomly dropped, the CDE's accuracy stays roughly constant while
  GRU-dt, GRU-D, GRU-ODE and ODE-RNN degrade (p.5).
- **Observational intensity** (sepsis, p.6): with it, CDE best but ODE-RNN and
  GRU-dt "both perform well"; without it "every model performs substantially
  worse", the benefit "particularly dramatic" for the CDE.
- ODE-RNN = GRU-dt plus a learnt ODE on the hidden state (p.5); GRU-dt is the
  standard discrete baseline, our LSTM+dt.

## What does not

- Supervised prediction, not control; no RL.
- The interpolated path needs future observations within the window; a causal
  controller cannot use the cubic-spline variant as is.

## Contact with our design

Related work, continuous-time paragraph, and the Limitations sentence on i.i.d.
gaps: the published gains lean on informative observation timing, which our
Tier 4 removes by design.
