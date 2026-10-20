---
id: doi-10.1371-journal.pcbi.1009418
title: "Drug-induced resistance evolution necessitates less aggressive treatment"
authors: "Kuosmanen T, Cairns J, Noble R, Beerenwinkel N, Mononen T, Mustonen V"
venue: "PLoS Computational Biology 2021;17(9):e1009418 (PMID 34555024, PMC8491903)"
peer_reviewed: yes
date_published: 2021
date_reviewed: "2026-10-20"
pdf_url: https://doi.org/10.1371/journal.pcbi.1009418
provenance: 1
status: reviewed
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
verdict_line: "Second MECHANISM (drug-induced resistance) read 1°; but the paper itself finds the optimum is an almost time-independent constant dose with only marginal closed-loop gain, and excludes standing variation (R0 = 0): little room for memory in the paper's own regime"
---

# Drug-induced resistance evolution necessitates less aggressive treatment

> **Provenance: `1°`** — read from `literature/pdf/doi-10.1371-journal.pcbi.1009418.pdf` (22 pp.) on 2026-10-20: abstract, Results, Discussion, Methods (pp.15-17, Eq. 6-9) and Table 1 (p.18), read from the rendered pages. **Not read:** the S1 Text derivations and the supplementary figures.

## The model (Eq. 7, p.16; Table 1, p.18)
    dS/dt = r_S (1 - S - R) S - d_max (1 - 1/(1 + (u/h)^k)) S - (mu_0 + alpha u) S,   S(0) = 1
    dR/dt = r_R (1 - S - R) R + (mu_0 + alpha u) S,                                    R(0) = 0
K = 1 scaled. Table 1: mu_0 = 1e-6, alpha = 1e-8 per ([u][t]), r_S = 0.5, r_R = 0.4, d_max = 1.0, u_max = 1e3, k = 2.3, h = 40, T = 35, K = 1e6 cells, S_0 = 1e5, R_0 = 0. Gillespie propensities in Eq. 8.

## What it says
- Cure probability is maximised at an intermediate CONSTANT dose; the drug raises the S->R rescue rate linearly with dose while the kill saturates (Hill). With dose-independent mutation the optimum reduces to MTD.
- Objective: elimination (probability of evolutionary rescue) over a fixed end-time, with the initial resistant population SET TO ZERO (de novo rescue only). Standing variation is explicitly a different regime that "shifts the focus towards containment".
- Closed loop: "Implementing the precise density and time dependencies lead only to marginal improvements" (Results); closed-loop protocols "hinge on continuous monitoring" and remain untested. The gain of the optimal constant dose over MTD depends on tolerability: 25-100% less mutation intensity well below the maximum dose for alpha = 10 (fold change < 10).

## Caveats for us
- Per-cell rates are 1e-6 to 1e-5 per time unit: in a density model with our pre-existing resistant fraction (1e-3 to 1e-1) the induced flux is negligible unless the slope is raised far beyond the paper's range.
- With R(0) = 0 and a constant optimal dose, a memoryless policy can already represent the optimum: this regime may have no room for architecture differences (LAW L1's headroom gate).
- Our score (TTP minus exposure, burden-only) is not the paper's objective.
