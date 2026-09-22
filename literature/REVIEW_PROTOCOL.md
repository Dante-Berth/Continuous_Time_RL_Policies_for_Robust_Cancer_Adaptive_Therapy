# Paper review protocol — continuous-time RL × mathematical oncology

How this program reads the literature. Written to be followed by someone who has
not read the spec.

**Where this sits in the precedence ladder: below everything.** A paper is not
evidence for this program. Only a logged, gated run on our own simulators is
([RESEARCH_LOG.md](../adaptive_therapy/RESEARCH_LOG.md)). A reviewed claim goes
into [PAPERS.md](PAPERS.md) and, if it changes a design choice, into a `/prereg`
entry — **never** directly into `VERDICTS.md`, `LAWS.md` or `CLOSED.md`. The
literature can tell us what to test and what to control for. It cannot tell us
what is true here.

**Why this subsystem exists at all.** [PROJECT.md](../PROJECT.md) opens with a
non-negotiable discipline rule, written because this project already produced one
AI-generated summary with *fabricated* citations — a phantom "late-2025 UDE
tumour paper" with a broken `[cite: topic-1.2.3]` placeholder, plus bare
source-tags standing in for references. This protocol makes that failure mode a
build error rather than a matter of vigilance.

| File | Kind | Changes |
|---|---|---|
| `REVIEW_PROTOCOL.md` (this file) | the stable spec — fields, tiers, rules | rarely, by discussion |
| `papers/<id>.md` | one file per paper, front-matter + prose — **the source of truth** | append-only per paper |
| [PAPERS.md](PAPERS.md) | the index — **generated**, never hand-edited | on regeneration |
| [../research/lit_table.py](../research/lit_table.py) | regenerates the index | — |

    uv run python research/lit_table.py --write    # regenerate
    uv run python research/lit_table.py --check    # exit 1 if stale
    make lit-check                                 # the same check

---

## 0. Mechanics

### 0.1 Layout and naming

```
literature/
  REVIEW_PROTOCOL.md      this file
  PAPERS.md               the index — GENERATED, do not hand-edit
  papers/
    doi-10.1158-0008-5472.CAN-23-0459.md    one file per paper, stubs included
  pdf/                    downloaded PDFs — gitignored, never committed
    <id>.pdf              named by paper id, so a review and its PDF pair up
```

ID scheme, in this order of preference:
`doi-<doi with / → ->` → `arxiv-NNNN.NNNNN` → `pmc-NNNNNNN` → `url-<domain>-<slug>`.
The ID is the filename, the table key, and how two readers refer to the same
paper. **Never renumber an ID.**

### 0.2 Status lifecycle

    queued -> screened -> reviewed
                       -> rejected (with a one-line reason, kept forever)

A `rejected` row is never deleted. The rejections are what make the index a
measurement rather than a highlight reel.

### 0.3 Provenance — the load-bearing rule

| Tier | Meaning |
|---|---|
| `1` (`1°`) | **Read from the PDF.** Every quantitative cell carries a page locator. |
| `2` (`2°`) | **Secondhand** — seeded from another summary, a citation in another paper, or an abstract. No PDF on disk. |

> **A `2°` cell may not be cited in a design argument.**

It may be cited as *"this direction exists"*. It may not be cited as *"X is
true, therefore we should build Y"*. Promoting a row to `1°` requires reading
the PDF and adding locators — nothing else.

The thirteen references verified in PROJECT.md §12-REF start at `2°`. They
resolve — that was checked by hand — but resolving is not the same as read. This
is the honest starting state, and it makes the reading queue generate itself.

### 0.4 Locators

Every quantitative claim carries `(<what>, p.N)` pointing at the PDF page that
holds it. A claim with no locator is not quotable in a design argument, even at
`1°`. If a number cannot be traced to a page, write `?` rather than the number.

---

## 1. Screen (before a full review)

Five questions, answerable from the abstract and the methods section:

| # | Field | Question |
|---|---|---|
| S1 | `object` | What does it actually produce — a policy, a model, a survey, a dataset? |
| S2 | `model_family` | Which tumour dynamics: `lotka-volterra`, `hahnfeldt`, `kuznetsov`, `other`, `none`? |
| S3 | `observability` | `burden-only`, `full-state`, or `n/a`? Full-state results do not transfer to our POMDP. |
| S4 | `endpoint` | `TTP`, `metastasis`, `OS`, `survival-proxy`, `none`. |
| S5 | `seeds` | How many seeds, and are variance bands reported? |

**Promotion trigger** — screen to full review if any of:
- it occupies one of the §3 lanes (direct competitive overlap);
- it bears on identifiability (`identifiability_contact: yes`);
- it reports a baseline we must beat or reproduce;
- it contradicts a standing design choice.

---

## 2. Front-matter schema

Every `papers/<id>.md` opens with a YAML block. Required fields:

```yaml
id: doi-10.1158-0008-5472.CAN-23-0459
title: "..."
authors: "..."
venue: "Cancer Research 2024;84(11):1929-1941"
peer_reviewed: yes            # yes | no | preprint
date_published: 2024
pdf_url: https://...
provenance: 2                 # 1 = PDF read, 2 = secondhand
status: queued                # queued | screened | reviewed | rejected
relevance_tier: 1             # 1 = direct competitor, ... 7 = background

object: policy                # policy | model | survey | dataset | theory
model_family: lotka-volterra
observability: burden-only    # burden-only | full-state | n/a
endpoint: TTP
seeds: "?"
identifiability_contact: no   # does it bear on PROJECT.md §5

route: mechanism              # mechanism | control | threshold | null | nothing
law_contact: none             # e.g. "L1 AGAINST", or none
headline: "one sentence, with a locator if 1°"
verdict_line: "what this paper does to our design, in one line"
```

### `route` — what the paper does to our design

| Value | Meaning |
|---|---|
| `mechanism` | changes what we model or how |
| `control` | names a confound we must control for |
| `threshold` | supplies a number we should adopt or beat |
| `null` | a negative result that closes a direction |
| `nothing` | context only; changes no design choice |

`route: nothing` rows are kept. They are most of any honest pile.

---

## 3. Review body

After the front-matter, prose under fixed headings:

    ## Screen (§1)        the five-row table
    ## What it does       the setup, in our vocabulary not theirs
    ## What transfers     the specific claim we could use, with locators
    ## What does not      assumptions that break under burden-only / multi-mechanism
    ## Contact with our design   which §-of-PROJECT.md it touches, and how

Keep it short. The front-matter is what the index reads; the prose is for the
person deciding whether to trust a number.
