---
id: url-zhang-2017-mcrpc-at
title: "Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer"
authors: "Zhang J, Cunningham JJ, Brown JS, Gatenby RA"
venue: "Nature Communications 2017;8:1816, doi 10.1038/s41467-017-01968-5 (PMID 29180633, PMC5703947)"
peer_reviewed: yes
date_published: 2017
date_reviewed: 2026-10-02
pdf_url: https://doi.org/10.1038/s41467-017-01968-5
provenance: 1
status: reviewed
relevance_tier: 2

object: model
model_family: other
observability: burden-only
endpoint: TTP
seeds: "n/a: two representative competition matrices (Table 3, p.8); trial n = 11 (p.4)"
identifiability_contact: no

route: threshold
law_contact: none
headline: "Three-type LV (T+, TP, T−) with producer-dependent capacities: abiraterone cuts K_TP 10,000 → 100 and K_T+ 1.5 → 0.5 x_TP (Eq. 1, p.7); PSA′ = Σx − 0.5 PSA (Eq. 2, p.8); AT stops at 50% of baseline PSA, resumes at baseline (p.8)"
verdict_line: "The candidate second family with resistance. Structure reproduces (22/22 SI equilibria within 1.9 pp); times do not from the main text (V5, L1176), and Table 3 Patient #1 contradicts the SI (V5d, L1216). No simulator adopted yet."
---

# Integrating evolutionary dynamics into treatment of mCRPC

> **Provenance: `1°`.** Read from `literature/pdf/url-zhang-2017-mcrpc-at.pdf`
> (9 pp.). Locators are article pages; `SI p.N` are pages of
> `url-zhang-2017-mcrpc-at-SI.pdf` (3 pp.: Supp. Tables 1–3, all 22 orderings).

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `model` plus a pilot trial of its rule (n = 11, p.4) |
| S2 | Model family | `other`: three-type LV competition with drug-dependent capacities (p.7) |
| S3 | Observability | `burden-only`: the protocol reads PSA (p.8) |
| S4 | Endpoint | TTP: radiographic progression; in the model, T− takes over with PSA above 50% of pre-treatment (p.8) |
| S5 | Seeds | n/a: deterministic; two representative matrices (Table 3, p.8) |

## What it does

dx_i/dt = r_i x_i (1 − Σ_j α_ij x_j / K_i) for T+ (androgen-dependent), TP
(testosterone-producing) and T− (androgen-independent) cells (Eq. 1, p.7).
r = 0.1 × (0.278, 0.355, 0.665) /day, scaled from cell-line doubling times
(p.7). K_T+ = 1.5 x_TP off drug, 0.5 x_TP on; K_TP = 10,000 off, 100 on; K_T− =
10,000 (p.7). α_ij: row i, column j; α_ii = 1 and 0 < α_ij < 1, with six
ordering rules (p.7); 22 orderings, valued 0.4–0.9 (p.7). Representative
patients in Table 3 (p.8). PSA′ = Σx − 0.5 PSA (Eq. 2, p.8). Start: equilibrium
frequencies at 25% of the untreated total; treat once PSA reaches 80% of
equilibrium (p.8). Table 1 (p.7): times to 5% and 90% T−; Patient #2 323 for
every regimen, MTD 784, adaptive "indefinite". Trial: median TTP ≥ 27 months on
47% of the standard cumulative dose (p.4).

## What transfers

A resistance mechanism structurally different from our two-type LV: the drug
acts on producer-dependent capacities, not on a growth term. PSA is a burden
observable with its own dynamics.

## What does not

The main text does not fix the time unit, how T− arises when absent at
equilibrium (Patient #1), or the metronomic schedule's durations (p.3, p.8).
From the main text alone, Patient #2's Table 1 times are off by ~7× (V5, L1176).
The SI fixes the structure but not the time unit, the integration scheme or the
T− seeding. Our equilibria match all 22 rows of Supp. Table 1 (SI p.1) within
1.9 pp, but Table 3's Patient #1 matrix (p.8) is SI row 22, not row 7 (SI p.3):
the paper contradicts itself there (V5d, L1216).

## Contact with our design

The AT50-style trial rule this program's thresholds descend from. As a second
simulator it needs the Supplementary Information and a fresh registration.
