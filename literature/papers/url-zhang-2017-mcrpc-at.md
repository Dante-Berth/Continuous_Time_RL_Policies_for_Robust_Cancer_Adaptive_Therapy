---
id: url-zhang-2017-mcrpc-at
title: "Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer"
authors: "Zhang J, Cunningham JJ, Brown JS, Gatenby RA"
venue: "Nature Communications 2017 (referenced by Strobl 2021)"
peer_reviewed: yes
date_published: 2017
date_reviewed: "?"
pdf_url: https://doi.org/10.1038/s41467-017-01968-5
provenance: 2
status: queued
relevance_tier: 2

object: policy
model_family: lotka-volterra
observability: burden-only
endpoint: TTP
seeds: "?"
identifiability_contact: no

route: threshold
law_contact: none
headline: "The clinical adaptive-therapy algorithm: dose on PSA rise, withdraw on PSA fall past a fraction of baseline"
verdict_line: "The AT algorithm itself -- the burden-threshold rule Gallagher's learned policy collapsed to. The interpretable structure Q1-alt asks whether a policy can preserve."
---

# Integrating evolutionary dynamics into treatment of mCRPC

> **Provenance: `2°` — NOT READ.** Seeded from the verified reference list in
> [PROJECT.md](../../PROJECT.md) §12-REF. The citation resolves; the paper has
> not been read from the PDF. **No cell in this file may be cited in a design
> argument** until it is promoted to `1°` (REVIEW_PROTOCOL.md §0.3).

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `policy` — a clinical protocol, not a learned one |
| S2 | Model family | `lotka-volterra` |
| S3 | Observability | `burden-only` (PSA) |
| S4 | Endpoint | `TTP` |
| S5 | Seeds | `?` — clinical trial, not a simulation study |

## What it does

*(pending — requires reading the PDF)*

## What transfers

*(pending — no locators until `1°`)*

## What does not

*(pending)*

## Contact with our design

The human-designed threshold rule that any learned policy is implicitly compared
against. PROJECT.md Q1-alt asks whether that interpretable structure survives
when the driving mechanism is unknown — this is the structure in question.
