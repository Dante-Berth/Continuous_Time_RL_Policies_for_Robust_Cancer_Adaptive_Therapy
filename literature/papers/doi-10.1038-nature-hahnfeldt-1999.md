---
id: doi-10.1038-nature-hahnfeldt-1999
title: "Tumor development under angiogenic signaling: a dynamical theory of tumor growth, treatment response, and postvascular dormancy"
authors: "Hahnfeldt P, Panigrahy D, Folkman J, Hlatky L"
venue: "Cancer Research 1999;59(19):4770-4775"
peer_reviewed: yes
date_published: 1999
date_reviewed: 2026-10-02
pdf_url: https://aacrjournals.org/cancerres/article/59/19/4770
provenance: 1
status: reviewed
relevance_tier: 2

object: model
model_family: hahnfeldt
observability: full-state
endpoint: none
seeds: "n/a: deterministic model; Monte-Carlo least-squares fits, ~1e6 runs (p.4770)"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "dV/dt = −λ1 V log(V/K), dK/dt = −λ2 K + bV − dKV^(2/3) − eKg(t) (Eqs. C1, C4, p.4771-4772); Lewis lung in mice: λ1 0.192, b 5.85, d 0.00873, K0 625, λ2 ≈ 0; drug in mg/kg with bolus PK (Table 1, p.4771)"
verdict_line: "Our vector field is this one; our drug scale was not. In its own units it reproduces 3 of 5 published targets, and the other 2 within 3.1–10.5% (V5, RESEARCH_LOG L1176). A mouse model with no resistant population: hosts a transfer test at most, never an adaptive-therapy comparison."
---

# Tumor development under angiogenic signaling: a dynamical theory of tumor growth, treatment response, and postvascular dormancy

> **Provenance: `1°`.** Read from `literature/pdf/doi-10.1038-nature-hahnfeldt-1999.pdf`
> (7 pp., Cancer Res 59:4770-4775; PMID 10519381). Locators are journal pages.
> **The id is wrong** — this is not a Nature paper and has no DOI in PubMed —
> but ids are never renumbered (REVIEW_PROTOCOL §0.1).

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `model`, fitted to Lewis lung carcinoma in C57BL/6 mice (p.4770) |
| S2 | Model family | `hahnfeldt` |
| S3 | Observability | `full-state` (caliper volumes; K is latent, p.4773) |
| S4 | Endpoint | none; tumour volume and set point |
| S5 | Seeds | n/a: deterministic; ~1e6 Monte-Carlo runs per fit (p.4770) |

## What it does

Volume V follows Gompertz growth toward a time-varying vascular capacity K:
dV/dt = −λ1 V log(V/K) (C1); dK/dt = −λ2 K + bV − dKV^(2/3) − eKg(t) (C4), with
the V^(2/3) term from a diffusion argument for inhibitors (p.4772). Drug enters
as concentration g(t) in mg/kg, boli decaying at clr (C3, p.4771; boli p.4773).
Table 1 (p.4771): λ1 0.192 /d, λ2 0, b 5.85 /d, d 0.00873 /d/vol^(2/3), K0 625
mm³; (e /d/conc, clr /d): TNP-470 (1.3, 10.1), endostatin (0.66, 1.7),
angiostatin (0.15, 0.38). Treatment starts at ≈200 mm³ (p.4772). Endostatin
and angiostatin 20 mg/kg/day control growth; endostatin 4 mg/kg/day and TNP-470
30 mg/kg q.o.d. do not (p.4773). Untreated set point ≈17,400 mm³; TNP-470 set
point ≈12,300 mm³ whatever the start; angiostatin 14 mg/kg/day overshoots, then
settles at ≈240 mm³ (p.4774, Fig. 4). TNP-470 at day 13: 1840 mm³ by boli vs
1300 continuous (p.4773).

## What transfers

The equations, which `envs/hahnfeldt.py` already implements (λ2 = 0). The
drug's scale: endostatin 20 mg/kg/day gives e·g ≈ 8 /day, about ten times the
e·u ≤ 0.8 /day our normalised PK allowed (W5-pre, L1004).

## What does not

The timescale: mouse tumours over days, daily boli, a horizon of ~2 weeks
(Fig. 1, p.4772). Our weekly-visit, one-year episode cannot host it as is.
There is no resistant population, so maximum dose is right by construction.
Patient-to-patient ranges are not given here (one fit).

## Contact with our design

W5-pre's "drug cannot outpace vascular growth" was our scaling, not this model
(V5, L1176). Hahnfeldt stays closed for dosing (CLOSED #3).
