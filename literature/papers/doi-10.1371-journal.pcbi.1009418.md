---
id: doi-10.1371-journal.pcbi.1009418
title: "Drug-induced resistance evolution necessitates less aggressive treatment"
authors: "Kuosmanen T, Cairns J, Noble R, Beerenwinkel N, Mononen T, Mustonen V"
venue: "PLoS Computational Biology 2021;17(9):e1009418 (PMID 34555024, PMC8491903)"
peer_reviewed: yes
date_published: 2021
date_reviewed: "?"
pdf_url: https://doi.org/10.1371/journal.pcbi.1009418
provenance: 2
status: queued
relevance_tier: 1

object: model
model_family: lotka-volterra
observability: full-state
endpoint: "probability of elimination (evolutionary rescue), not TTP"
seeds: "n/a (optimal control; Gillespie simulation, 2000 runs per dose)"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "Sensitive S and resistant R cells with Hill pharmacodynamics and a S->R rescue-mutation rate that rises linearly with dose; the probability of cure is maximised at an INTERMEDIATE constant dose, below the dose of maximal decay (abstract and prose)"
verdict_line: "W14 candidate (second MECHANISM: drug-induced rather than selected resistance); prose read 2026-10-20 via PMC, equations and Table 1 values were lost in text extraction, so the PDF must be read before any parameter is used"
---

# Drug-induced resistance evolution necessitates less aggressive treatment

> **Provenance: `2°`** — abstract and the full prose from PMC8491903 (2026-10-20). **The text layer dropped every equation and the numbers of Table 1.** Not usable for parameters or a design argument until the PDF is read.

## What it says (prose only)
- Model: sensitive and resistant cells, each a birth-death process; the drug kills sensitive cells (Hill pharmacodynamics, finite leverage); a rescue-mutation rate S->R increases with dose (linear). Resistant cells are not affected by the drug and carry a fitness-cost option. Methods: Pontryagin and HJB optimal control, plus Gillespie simulations.
- Result: a trade-off between decreasing the sensitive population and generating rescue mutants; cure probability peaks at an intermediate dose. With dose-independent mutation the optimum reduces to MTD (Discussion).
- Objective is elimination over a fixed end-time with the initial resistant population set to zero, not time to progression and not a pre-existing resistant fraction.

## Caveats for us
- Our score (TTP minus exposure, burden-only observation) is not their objective; any adaptation is ours.
- The paper chooses its mutation-slope parameter so the dose-dependent rate "covers an order of magnitude"; a pre-registered grid must follow that, not a value picked for a result.
