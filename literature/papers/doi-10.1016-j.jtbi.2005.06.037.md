---
id: doi-10.1016-j.jtbi.2005.06.037
title: "Mixed immunotherapy and chemotherapy of tumors: modeling, applications and biological interpretations"
authors: "de Pillis LG, Gu W, Radunskaya AE"
venue: "Journal of Theoretical Biology 2006;238(4):841-862 (PMID 16153659)"
peer_reviewed: yes
date_published: 2005
date_reviewed: "2026-10-19"
pdf_url: https://doi.org/10.1016/j.jtbi.2005.06.037
provenance: 1
status: reviewed
relevance_tier: 1

object: model
model_family: other
observability: full-state
endpoint: "none (tumour eradicated or at carrying capacity; qualitative time courses)"
seeds: "n/a (deterministic ODEs)"
identifiability_contact: no

route: mechanism
law_contact: none
headline: "6-state ODE (tumour T, NK N, CD8+ L, lymphocytes C, chemo M, IL-2 I); chemo kills tumour AND immune cells (1-e^-M fractional kill, K_N=K_L=K_C < K_T, Table 1); mouse set and two human patients (Table 2); tumour-free equilibrium unstable, high-tumour stable (p.857)"
verdict_line: "W13 candidate, read 1°; gated by PREREG W13 [2026-10-19] (reproduction of Figs 6-8, then headroom) before any policy use"
---

# Mixed immunotherapy and chemotherapy of tumors: modeling, applications and biological interpretations

> **Provenance: `1°`** — read from `literature/pdf/doi-10.1016-j.jtbi.2005.06.037.pdf` (22 pp.) on 2026-10-19: the model (§2, Eqs. 1-8), parameter derivation (§3), numerical experiments (§5-6) and discussion (§8), and Tables 1-2. **Not read in detail:** the equilibrium and bifurcation analysis (§4) and the vaccine experiments (§7). Locators are journal pages.

## What it says
- Eqs. 1-8 (pp.845-846): tumour logistic growth; NK and CD8+ killing with saturating recruitment and inactivation; chemotherapy kill `K_f (1 - e^-M)` on **T, N, L and C**; circulating lymphocytes C feed NK production, so chemo suppresses immunity indirectly (p.844). Chemo is a bolus `v_M(t)` decaying at rate `γ` (half-life ~18 h, p.846).
- p.846: K_N, K_L, K_C are "smaller than K_T, but of the same order of magnitude". p.850: "Heavy chemotherapy will harm other tissues"; the combination at lower doses is argued to be gentler.
- No resistant compartment. The mechanism is immune escape: for the mouse set the tumour-free equilibrium is unstable and a high-tumour one is stable, so outcomes depend sharply on initial conditions (p.857, Fig. 6).
- Schedule sensitivity (p.851, Fig. 8): for a 2e7-cell human-parameter tumour, chemo every 5 days eradicates it by day ~50; the same total drug every 10 days lets it regrow.
- Fig. 6 (mouse, T0 = 1e6): no treatment reaches capacity; 7 chemo pulses (1 day in a 14-day cycle) do not control it; combination chemo + TIL does. Fig. 7: patient 9 with a healthy immune state controls a 1e6 tumour, with a depleted state it does not.
- The authors say the time scale of the combination-therapy tumour decline "may not reflect realistic dynamics" (p.850).

## Caveats for us
- Table exponents did not survive text extraction in this review; **take every value from the paper's tables, not from this note.**
- Validation is qualitative against two data sets, not a fit. Human parameters are two patients.
- It is not an adaptive-therapy model: there is no TTP; the outcome is eradication or escape.
