---
id: doi-10.1158-0008-5472.CAN-23-0713
title: "Deep reinforcement learning for adaptive therapy in prostate cancer"
authors: "Gallagher K, Strobl MAR, Park DS, Spoendlin FC, Gatenby RA, Maini PK, Anderson ARA"
venue: "Cancer Research 2024;84(11):1929-1941"
peer_reviewed: yes
date_published: 2024
date_reviewed: "?"
pdf_url: https://doi.org/10.1158/0008-5472.CAN-23-0713
provenance: 2
status: queued
relevance_tier: 1

object: policy
model_family: lotka-volterra
observability: burden-only
endpoint: TTP
seeds: "?"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "A3C DRL on an LV prostate model; reducing monitoring/decision frequency increases time-to-progression"
verdict_line: "The direct baseline and the source of Q1. Same environment, same A3C setup -- we test whether its frequency finding is mechanism-independent."
---

# Deep reinforcement learning for adaptive therapy in prostate cancer

> **Provenance: `2°` — NOT READ.** Seeded from the verified reference list in
> [PROJECT.md](../../PROJECT.md) §12-REF. The citation resolves; the paper has
> not been read from the PDF. **No cell in this file may be cited in a design
> argument** until it is promoted to `1°` (REVIEW_PROTOCOL.md §0.3).

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `policy` |
| S2 | Model family | `lotka-volterra` |
| S3 | Observability | `burden-only` |
| S4 | Endpoint | `TTP` |
| S5 | Seeds | `?` — not established without the PDF |

## What it does

*(pending — requires reading the PDF)*

## What transfers

*(pending — no locators until `1°`)*

## What does not

*(pending)*

## Contact with our design

The direct baseline and the source of Q1. Same environment, same A3C setup -- we test whether its frequency finding is mechanism-independent.
