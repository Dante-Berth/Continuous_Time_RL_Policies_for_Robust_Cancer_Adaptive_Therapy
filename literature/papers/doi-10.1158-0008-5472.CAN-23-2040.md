---
id: doi-10.1158-0008-5472.CAN-23-2040
title: "Mathematical Model-Driven Deep Learning Enables Personalized Adaptive Therapy"
authors: "Gallagher K, Strobl MAR, Park DS, Spoendlin FC, Gatenby RA, Maini PK, Anderson ARA"
venue: "Cancer Research 2024;84(11):1929-1941"
peer_reviewed: yes
date_published: 2024
date_reviewed: 2026-09-24
pdf_url: https://doi.org/10.1158/0008-5472.CAN-23-2040
provenance: 1
status: reviewed
relevance_tier: 1

object: policy
model_family: lotka-volterra
observability: burden-only
endpoint: TTP
seeds: "not stated; means over 100 stochastic evaluations (p.1933)"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "A3C on burden-only LV learns a sigmoidal treat-above-N* switch, N* = 0.83 n0 (Fig. 4B, p.1935); TTP rises when decisions are LESS frequent in training (Fig. 3A, p.1934), but falls when evaluation is less frequent than training (Fig. 3C, p.1934)"
verdict_line: "Our setting, pre-occupied: LV, burden-only, TTP = +20%. Its threshold rule is the one our MLP learned (W1, ~0.80x baseline). Fig. 3C predicts W4's direction for any policy trained weekly and evaluated with dropped visits."
---

# Mathematical Model-Driven Deep Learning Enables Personalized Adaptive Therapy

> **Provenance: `1°`.** Read from
> `literature/pdf/doi-10.1158-0008-5472.CAN-23-2040.pdf` (13 pp., open access).
> Page numbers are the journal's (1929–1941), printed on each page.
>
> Supersedes the `2°` stub `doi-10.1158-0008-5472.CAN-23-0713`, which carried a
> wrong DOI and title; the PDF's own DOI is 10.1158/0008-5472.CAN-23-2040
> (p.1929).

## Screen (§1)

| # | Field | Value |
|---|---|---|
| S1 | Object | `policy` — A3C, input = current tumour size only (p.1932) |
| S2 | Model family | `lotka-volterra`, sensitive/resistant, shared K (Eq. A, p.1931) |
| S3 | Observability | `burden-only`, by design (p.1932) |
| S4 | Endpoint | TTP, progression = 20% growth from N₀ (p.1932) |
| S5 | Seeds | training seeds not stated; **means** over 100 stochastic evaluations (p.1933) |

## What it does

Binary dosing (treat / holiday) on a Lotka–Volterra S/R model with parameters
from Strobl et al. fits to Bruchovsky intermittent-ADT patients (p.1931).
Reward: +0.1 per step survived, +0.05 per holiday step, −0.1 at progression,
+5 at 30 years; γ = 0.999 (Table 1 and p.1933). Benchmarks: continuous therapy
and AT50 (p.1932).

## What transfers

- **Learned policy = a burden threshold.** Treatment probability is sigmoidal in
  current size, treat almost certainly above N* = 0.83 n₀ (Fig. 4B, p.1935);
  N* depends on the decision interval (Fig. 4C, p.1935); personalised
  thresholds span 73–100% of n₀ (p.1937).
- **Decision frequency.** Training with less frequent decisions raises TTP
  (Fig. 3A, p.1934; P < 0.05 pairwise, p.1933). Evaluating LESS often than
  trained lowers TTP ("premature progression"; Fig. 3C, p.1934, P = 0.035
  weekly-trained). Evaluating more often than trained: no mean change.
- **Robustness.** Single-patient policy is robust to initial resistance and
  resistance cost, but very sensitive to reduced turnover and initial density,
  i.e. faster-than-trained growth (p.1935; Supp. Fig. S2). Cohort training
  trades specialisation for robustness (Fig. 5C–D, p.1936) and still fails
  outside the training space (p.1937).
- **Missed visits.** Exponentially distributed decision delays: single-patient
  model loses ≥ 20% TTP at mean delay < 1 week; cohort model tolerates ~1 month
  (p.1937; Supp. Fig. S5).

## What does not

- Means and 95% CIs over stochastic evaluations of one trained network, not
  medians across seeds; no paired test across a matched cohort.
- Exposure is reported as a side benefit (−50% cumulative dose, p.1937), not
  as a matched axis — a TTP comparison here is not exposure-controlled.
- Binary dosing; our action set is {0, ½, 1}.
- Mechanism shift is tested (Lu model, stem-cell model; p.1936), but with the
  progression criterion changed too, so it is not a clean Tier-2 analogue.

## Contact with our design

- W1 ([L461](../../adaptive_therapy/RESEARCH_LOG.md#L461)): our PQN MLP
  independently learned the same kind of rule, dose off below ~0.80× baseline.
  This is a sanity check on our simulator and trainer, not a novelty.
- W4 (Tier 4): Fig. 3C and the delay result (p.1937) predict that dropping
  visits at test time hurts a weekly-trained policy through premature
  progression. The W4 prereg must say what dosing continues across a missed
  visit, since that decides whether the tumour can outrun the rule.
- Stage 2 / Q1: its frequency finding is on one LV model; whether it holds
  across mechanisms remains our open question.
