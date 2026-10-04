---
id: url-neurips-2019-latent-odes
title: "Latent ODEs for Irregularly-Sampled Time Series"
authors: "Rubanova Y, Chen RTQ, Duvenaud D"
venue: "NeurIPS 2019, pp. 5320-5330 (arXiv:1907.03907v1)"
peer_reviewed: yes
date_published: 2019
date_reviewed: 2026-10-04
pdf_url: https://papers.nips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html
provenance: 1
status: reviewed
relevance_tier: 2

object: model
model_family: none
observability: n/a
endpoint: none
seeds: "see source tables"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "ODE-RNN: hidden state follows a learnt ODE between observations and jumps by an RNN cell at them (Alg. 1, p.3); better than RNNs when data are sparse, and observation absence can itself be informative (p.2)"
verdict_line: "Our ODE-RNN arm is Alg. 1 (GRU jump + learnt flow). The paper's evidence is supervised and leans on sparse/informative timing, which i.i.d. missed visits do not supply."
---

# Latent ODEs for Irregularly-Sampled Time Series

> **Provenance: `1°`.** Read from `literature/pdf/url-neurips-2019-latent-odes.pdf`. Page numbers
> are the PDF's own.

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `model` |
| S2 | Model family | `none` |
| S3 | Observability | `n/a` |
| S4 | Endpoint | `none` |
| S5 | Seeds | see source tables |

## What it does

Generalises RNN state transitions to a neural-ODE flow between observations
(ODE-RNN), and uses the ODE-RNN as the encoder of a latent ODE VAE; optionally
models observation times with a Poisson process (abstract, p.1).

## What transfers

- **Alg. 1** (p.3): h'_i = ODESolve(f, h_{i-1}, (t_{i-1}, t_i)); h_i = RNNCell(h'_i, x_i).
  Our arm follows this, with a GRU cell and a tanh-bounded flow.
- "ODE-RNNs can perform better when the data is sparse"; "the absence of
  observations itself can be informative" (p.2).
- Motivation: binning destroys timing information that "can be informative
  about latent variables" (p.1).

## What does not

- Interpolation/extrapolation and classification (MuJoCo, PhysioNet, human
  activity); no control or RL.

## Contact with our design

The architecture of our continuous-time arm and the claim we test. Related
work, continuous-time paragraph.
