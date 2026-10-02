# PAPERS — the living index

One row per paper, per [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md). This is the
index; the judgment lives in `papers/<id>.md`.

> **The tables below are GENERATED. Do not edit them by hand** — edit the
> front-matter of `papers/<id>.md` and regenerate:
>
> ```
> uv run python research/lit_table.py --write   # regenerate
> uv run python research/lit_table.py --check   # exit 1 if stale
> ```

**Provenance status: 6/17 read from the PDF (`1°`); 11 are `2°` (65%).**

A `2°` row is seeded from PROJECT.md §12-REF — the citation resolves, but
the paper has not been read. **A `2°` cell may not be cited in a design**
**argument** (REVIEW_PROTOCOL.md §0.3). It may be cited as *"this direction
exists"*, never as *"X is true, therefore build Y"*.

`?` = the source does not say · `—` = not applicable.

## What this pile changes

Every row whose `route` is not `nothing`, ordered `null` > `threshold` >
`control` > `mechanism`. This is the answer to "what does the literature do
to our design", and it is the only section most readers need.

| Route | Paper | Prov | What it does to the design |
|---|---|---|---|
| `threshold` | [pmc-8455086](papers/pmc-8455086.md) | 1° | Source of our 'strobl2021' LV ranges, now read 1°: every value matches Table 1 (p.29). Those are theory sweep ranges; its fitted patients reach costs well above 50% (Fig. 5C, p.26), so our provisional high-cost box is outside the sweep, not outside the patients. |
| `threshold` | [url-zhang-2017-mcrpc-at](papers/url-zhang-2017-mcrpc-at.md) | 1° | The candidate second family with resistance. Structure reproduces (22/22 SI equilibria within 1.9 pp); times do not from the main text (V5, L1176), and Table 3 Patient #1 contradicts the SI (V5d, L1216). No simulator adopted yet. |
| `threshold` | [doi-10.1371-journal.pone.0120007](papers/doi-10.1371-journal.pone.0120007.md) | 2° | Where the Hahnfeldt numbers come from. Pull from the source tables; do not hardcode from memory. |
| `threshold` | [doi-10.1155-2017-1642976](papers/doi-10.1155-2017-1642976.md) | 2° | Secondary source for the non-dimensional Kuznetsov values. |
| `control` | [arxiv-2605.02552](papers/arxiv-2605.02552.md) | 1° | Occupies memory-helps-under-POMDP, but on a single fixed patient with a hidden toxicity state, means over 10 seeds, and no distribution shift. Our W1 memoryless MLP beat the recurrent arm; its result does not predict ours. It sets the minimum baseline: an LSTM arm must exist (W3). |
| `control` | [arxiv-2608.18387](papers/arxiv-2608.18387.md) | 2° | The identifiability landmine. Forces Q3 to be robustness UNDER non-identifiability, and forces a second endpoint alongside TTP. |
| `control` | [arxiv-2609.12264](papers/arxiv-2609.12264.md) | 2° | Assumes full subpopulation observation (their stated limitation). We are burden-only, which is the harder and more realistic setting. |
| `control` | [url-iclr-2024-metaphysica](papers/url-iclr-2024-metaphysica.md) | 2° | A strong baseline we should be willing to lose a tier to. PROJECT.md 13 names beating only weak baselines as a reviewer reflex. |
| `mechanism` | [doi-10.1158-0008-5472.CAN-23-0713](papers/doi-10.1158-0008-5472.CAN-23-0713.md) | 2° | Superseded by doi-10.1158-0008-5472.CAN-23-2040. Kept, never deleted, per REVIEW_PROTOCOL §0.2. |
| `mechanism` | [doi-10.1158-0008-5472.CAN-23-2040](papers/doi-10.1158-0008-5472.CAN-23-2040.md) | 1° | Our setting, pre-occupied: LV, burden-only, TTP = +20%. Its threshold rule is the one our MLP learned (W1, ~0.80x baseline). Fig. 3C predicts W4's direction for any policy trained weekly and evaluated with dropped visits. |
| `mechanism` | [doi-10.1007-BF02460644](papers/doi-10.1007-BF02460644.md) | 2° | The third family. Produces dynamics LV and Hahnfeldt cannot, and the Hopf bifurcations that break a naive TTP rule (PROJECT.md 6.3). |
| `mechanism` | [doi-10.1038-nature-hahnfeldt-1999](papers/doi-10.1038-nature-hahnfeldt-1999.md) | 1° | Our vector field is this one; our drug scale was not. In its own units it reproduces 3 of 5 published targets, and the other 2 within 3.1–10.5% (V5, RESEARCH_LOG L1176). A mouse model with no resistant population: hosts a transfer test at most, never an adaptive-therapy comparison. |
| `mechanism` | [url-neurips-2019-latent-odes](papers/url-neurips-2019-latent-odes.md) | 2° | The encoder for the MoE latent model AND the Q4 irregular-sampling baseline the pure-liquid policy must beat. Do not reinvent it. |
| `mechanism` | [doi-10.1126-scirobotics.adc8892](papers/doi-10.1126-scirobotics.adc8892.md) | 2° | The OOD-robustness claim for continuous-time policies, in another domain. The architecture we are porting into oncology. |
| `mechanism` | [arxiv-2407.04811](papers/arxiv-2407.04811.md) | 1° | Adopted as the Phase 2c trainer for BOTH arms after PPO left the ODE-RNN open-loop (L312, L391); its stability theorem is a wide-network result, which our 16-unit ODE hidden state is not. |

## Read next

`2°` rows ranked by relevance tier. Reading a PDF and adding locators is
what promotes a row to `1°` and makes it citable in a design argument.

| Tier | Paper | Why it matters |
|---|---|---|
| 1 | [arxiv-2608.18387](papers/arxiv-2608.18387.md) | r_R is weakly identifiable under adaptive therapy; parameters fit under one schedule do not transport to another; TTP and metastasis risk pull against each other |
| 1 | [doi-10.1158-0008-5472.CAN-23-0713](papers/doi-10.1158-0008-5472.CAN-23-0713.md) | REJECTED: mis-seeded stub. Wrong DOI and title for Gallagher 2024; the PDF's own DOI is 10.1158/0008-5472.CAN-23-2040 (p.1929) |
| 2 | [arxiv-2609.12264](papers/arxiv-2609.12264.md) | TD3 vs DQN under parametric tumour heterogeneity, full observation, 100-patient generalization |
| 2 | [doi-10.1007-BF02460644](papers/doi-10.1007-BF02460644.md) | Effector-tumour predation dynamics with immunostimulation, sneaking-through, and a dormant state; BCL1 lymphoma parameters |
| 2 | [url-neurips-2019-latent-odes](papers/url-neurips-2019-latent-odes.md) | ODE-RNN: hidden state obeys an ODE between observations and jumps at them |
| 3 | [doi-10.1038-s41540-023-00317-1](papers/doi-10.1038-s41540-023-00317-1.md) | Neural-ODE tumour dynamic model with an interpretable latent space, predicting overall survival |
| 3 | [doi-10.1126-scirobotics.adc8892](papers/doi-10.1126-scirobotics.adc8892.md) | Liquid time-constant networks hold up out-of-distribution where discrete policies degrade |
| 3 | [doi-10.1371-journal.pone.0120007](papers/doi-10.1371-journal.pone.0120007.md) | Parameter ranges for the Hahnfeldt angiogenesis model |
| 3 | [url-iclr-2024-metaphysica](papers/url-iclr-2024-metaphysica.md) | Causal structure discovery plus meta-learning for OOD robustness in physics-informed ML |
| 4 | [doi-10.1155-2017-1642976](papers/doi-10.1155-2017-1642976.md) | Reuses the Kuznetsov non-dimensional parameter values in a delay setting |
| 4 | [url-neurips-2023-dlde-dnode](papers/url-neurips-2023-dlde-dnode.md) | Neural-ODE model of breast cancer tumour dynamics and progression-free survival |

## All rows

| ID | Venue | Obj | Family | Obs | Endpoint | Ident | Prov | Status |
|---|---|---|---|---|---|---|---|---|
| [arxiv-2605.02552](papers/arxiv-2605.02552.md) | arXiv:2605.02552v1, 4 May 2026 | policy | other | full-state | survival-proxy | no | 1° | reviewed |
| [arxiv-2608.18387](papers/arxiv-2608.18387.md) | arXiv:2608.18387 | theory | lotka-volterra | burden-only | metastasis | yes | 2° | queued |
| [doi-10.1158-0008-5472.CAN-23-0713](papers/doi-10.1158-0008-5472.CAN-23-0713.md) | Cancer Research 2024;84(11):1929-1941 | policy | lotka-volterra | burden-only | TTP | no | 2° | rejected |
| [doi-10.1158-0008-5472.CAN-23-2040](papers/doi-10.1158-0008-5472.CAN-23-2040.md) | Cancer Research 2024;84(11):1929-1941 | policy | lotka-volterra | burden-only | TTP | no | 1° | reviewed |
| [arxiv-2609.12264](papers/arxiv-2609.12264.md) | arXiv:2609.12264 | policy | other | full-state | survival-proxy | no | 2° | queued |
| [doi-10.1007-BF02460644](papers/doi-10.1007-BF02460644.md) | Bulletin of Mathematical Biology 1994;56(2):295-321 | model | kuznetsov | full-state | none | no | 2° | queued |
| [doi-10.1038-nature-hahnfeldt-1999](papers/doi-10.1038-nature-hahnfeldt-1999.md) | Cancer Research 1999;59(19):4770-4775 | model | hahnfeldt | full-state | none | no | 1° | reviewed |
| [pmc-8455086](papers/pmc-8455086.md) | Cancer Research 2021;81(4):1135-1147, doi 10.1158/0008-5472.CAN-20-0806 | model | lotka-volterra | n/a | TTP | yes | 1° | reviewed |
| [url-neurips-2019-latent-odes](papers/url-neurips-2019-latent-odes.md) | NeurIPS 2019, pp. 5320-5330 | model | none | n/a | none | no | 2° | queued |
| [url-zhang-2017-mcrpc-at](papers/url-zhang-2017-mcrpc-at.md) | Nature Communications 2017;8:1816, doi 10.1038/s41467-017-01968-5 (PMID 29180633, PMC5703947) | model | other | burden-only | TTP | no | 1° | reviewed |
| [doi-10.1038-s41540-023-00317-1](papers/doi-10.1038-s41540-023-00317-1.md) | npj Systems Biology and Applications 2023 | model | other | burden-only | OS | no | 2° | queued |
| [doi-10.1126-scirobotics.adc8892](papers/doi-10.1126-scirobotics.adc8892.md) | Science Robotics 2023 | policy | none | n/a | none | no | 2° | queued |
| [doi-10.1371-journal.pone.0120007](papers/doi-10.1371-journal.pone.0120007.md) | PLoS One 2015;10(3):e0120007 (PMC4364928) | model | hahnfeldt | full-state | none | no | 2° | queued |
| [url-iclr-2024-metaphysica](papers/url-iclr-2024-metaphysica.md) | ICLR 2024 | model | none | n/a | none | no | 2° | queued |
| [arxiv-2407.04811](papers/arxiv-2407.04811.md) | arXiv:2407.04811v2, 23 Oct 2024 (PDF header: under review at ICLR 2025) | theory | none | n/a | none | no | 1° | reviewed |
| [doi-10.1155-2017-1642976](papers/doi-10.1155-2017-1642976.md) | Wiley / Computational and Mathematical Methods in Medicine 2017 | model | kuznetsov | full-state | none | no | 2° | queued |
| [url-neurips-2023-dlde-dnode](papers/url-neurips-2023-dlde-dnode.md) | NeurIPS 2023 Workshop (DLDE) | model | other | burden-only | OS | no | 2° | queued |

## What the pile looks like

- **Model families:** `hahnfeldt` 2, `kuznetsov` 2, `lotka-volterra` 4, `none` 4, `other` 5
- **Observability:** `burden-only` 6, `full-state` 6, `n/a` 5
- **Bearing on identifiability (PROJECT.md §5):** 2

*Generated 2026-10-02 by `research/lit_table.py`.*
