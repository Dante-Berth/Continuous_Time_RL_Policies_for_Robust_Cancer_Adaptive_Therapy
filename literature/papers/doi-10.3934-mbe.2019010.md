---
id: doi-10.3934-mbe.2019010
title: "Dynamics and implications of models for intermittent androgen suppression therapy"
authors: "Phan T, He C, Martinez A, Kuang Y"
venue: "Mathematical Biosciences and Engineering 2019;16(1):187-204 (PMID 30674116)"
peer_reviewed: yes
date_published: 2019
date_reviewed: "2026-10-04"
pdf_url: https://doi.org/10.3934/mbe.2019010
provenance: 1
status: reviewed
relevance_tier: 1

object: model
model_family: other
observability: burden-only
endpoint: "fit and forecast RMSE, PSA and androgen (no TTP)"
seeds: "n/a (deterministic fmincon, hand-chosen initial guesses)"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "2-pop Droop (Baez-Kuang) fits 26 Vancouver IAS patients with median RMSE PSA 1.14, androgen 2.68 (Table 2, p.197); forecasts better than the 3-pop model (PSA 4.52 vs 5.32, Table 3, p.199)"
verdict_line: "ROADMAP V6 candidate, read 1°; gated by PREREG V6-gate [2026-10-04] before any policy use"
---

# Dynamics and implications of models for intermittent androgen suppression therapy

> **Provenance: `1°`** — read from the PDF 2026-10-04 (`literature/pdf/doi-10.3934-mbe.2019010.pdf`).

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `model` |
| S2 | Model family | `other` — cell-quota (Droop) growth, androgen-limited |
| S3 | Observability | PSA + serum androgen (testosterone) |
| S4 | Endpoint | fit/forecast RMSE; no TTP, no dosing optimisation |
| S5 | Seeds | n/a — `fmincon`, initial guesses "chosen by hand" (p.193) |

## What it does

- Two models: Baez–Kuang 2-population (AD x1 → AI x2, irreversible, rate
  λ(Q) = cK/(Q+K); eqs. 2.1–2.6, p.189) and a 3-population extension with a
  reversible weak-AI type (eqs. 2.7–2.13, p.190–191). Treatment is binary:
  γ = γ1·u + γ2, u = 0 on treatment (eq. 2.6, p.189).
- Ranges: Table 1 (p.192); γ1 = 20 fixed, several lower bounds are 0.
- Fit: MSE_PSA + MSE_androgen over the first 1.5 cycles, `fmincon` within
  Table 1 (p.193). Forecast: one more on/off cycle with the data's switch times (p.197).
- Cohort: Vancouver data, 109 patients; ≥ 20 PSA and androgen points in 1.5
  cycles ⇒ 62; 26 with a forecastable cycle (p.189). The 26 IDs are not listed.
- Results: fit RMSE median [Q1, Q3], 2-pop: PSA 1.14 [0.78, 2.04], androgen
  2.68 [1.88, 4.50]; 3-pop PSA 1.18 [0.69, 2.11], androgen identical (Table 2,
  p.197). Forecast: 2-pop PSA 4.52 [2.94, 8.10], androgen 4.19 [2.31, 6.62];
  3-pop PSA 5.32 [3.76, 15.61] (Table 3, p.199).
- Analysis: CAS relapse needs an absorbing (irreversible) state (Sec. 3.1, p.194).

## What transfers

- A resistance mechanism unlike LV: no pre-existing clone; resistance is
  created by deprivation (λ rises as Q falls). Validatable on data we hold.
- Androgen is a second observed channel, so the fit is less under-determined than PSA alone.

## What does not

- The stated cohort rule is not reproducible from our files: it admits 43/72
  of ours (literal "on, off, on"), and example patient 100 has 16 PSA / 11
  androgen points in that window, below the rule (Fig. 2, p.198).
- Hand-chosen initial guesses: the reported RMSEs are not a deterministic target.
- Treatment-onset delay of PSA relapse is noted but not modelled (A2, p.200).
- Table 2's androgen RMSE is identical across both models to 2 d.p.: unexplained.

## Contact with our design

ROADMAP V6 step 2: our implementation must reach Table 2–3 RMSE on our
patients before any policy uses it (PREREG V6-gate [2026-10-04]).
