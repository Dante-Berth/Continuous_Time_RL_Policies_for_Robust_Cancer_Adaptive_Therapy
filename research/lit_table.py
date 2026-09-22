#!/usr/bin/env python3
"""Regenerate literature/PAPERS.md from the front-matter of literature/papers/*.md.

The index is GENERATED. `papers/<id>.md` is the source of truth; PAPERS.md is a
decision view over it. Hand-editing the index creates a second copy that nobody
updates and that every later reader takes as true -- which is the accretion this
whole subsystem exists to prevent.

The index is decision-first. Its top sections are derived from the `route` field
and the provenance tier, not written by hand, so they cannot drift from the
rules that produced them.

Usage:
    uv run python research/lit_table.py --write     # regenerate
    uv run python research/lit_table.py --check     # exit 1 if stale
    uv run python research/lit_table.py --stdout    # print, do not write
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO / "literature" / "papers"
INDEX = REPO / "literature" / "PAPERS.md"

ROUTE_ORDER = ["null", "threshold", "control", "mechanism", "nothing"]


def parse_front_matter(path: Path) -> dict[str, str]:
    """Minimal YAML front-matter reader: flat `key: value` pairs only."""
    text = path.read_text()
    if not text.startswith("---"):
        return {}
    _, block, _ = text.split("---", 2)
    out: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"')
    return out


def load() -> list[dict[str, str]]:
    rows = [parse_front_matter(p) for p in sorted(PAPERS_DIR.glob("*.md"))]
    return [r for r in rows if r.get("id")]


def _cell(row: dict[str, str], key: str, default: str = "?") -> str:
    return row.get(key, default) or default


def render(rows: list[dict[str, str]]) -> str:
    n = len(rows)
    first = sum(1 for r in rows if _cell(r, "provenance") == "1")
    second = n - first
    unread_pct = round(100 * second / n) if n else 0

    L: list[str] = []
    A = L.append

    A("# PAPERS — the living index")
    A("")
    A("One row per paper, per [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md). This is the")
    A("index; the judgment lives in `papers/<id>.md`.")
    A("")
    A("> **The tables below are GENERATED. Do not edit them by hand** — edit the")
    A("> front-matter of `papers/<id>.md` and regenerate:")
    A(">")
    A("> ```")
    A("> uv run python research/lit_table.py --write   # regenerate")
    A("> uv run python research/lit_table.py --check   # exit 1 if stale")
    A("> ```")
    A("")
    A(
        f"**Provenance status: {first}/{n} read from the PDF (`1°`); "
        f"{second} are `2°` ({unread_pct}%).**"
    )
    A("")
    if second:
        A("A `2°` row is seeded from PROJECT.md §12-REF — the citation resolves, but")
        A("the paper has not been read. **A `2°` cell may not be cited in a design**")
        A('**argument** (REVIEW_PROTOCOL.md §0.3). It may be cited as *"this direction')
        A('exists"*, never as *"X is true, therefore build Y"*.')
        A("")
    A("`?` = the source does not say · `—` = not applicable.")
    A("")

    # --- Section 1: what this pile changes -------------------------------
    A("## What this pile changes")
    A("")
    A("Every row whose `route` is not `nothing`, ordered `null` > `threshold` >")
    A('`control` > `mechanism`. This is the answer to "what does the literature do')
    A('to our design", and it is the only section most readers need.')
    A("")
    A("| Route | Paper | Prov | What it does to the design |")
    A("|---|---|---|---|")
    routed = [r for r in rows if _cell(r, "route", "nothing") != "nothing"]
    routed.sort(
        key=lambda r: (
            ROUTE_ORDER.index(_cell(r, "route", "nothing"))
            if _cell(r, "route", "nothing") in ROUTE_ORDER
            else 99,
            int(_cell(r, "relevance_tier", "9") or 9),
        )
    )
    for r in routed:
        prov = "1°" if _cell(r, "provenance") == "1" else "2°"
        A(
            f"| `{_cell(r, 'route')}` | [{r['id']}](papers/{r['id']}.md) "
            f"| {prov} | {_cell(r, 'verdict_line', '—')} |"
        )
    A("")

    # --- Section 2: read next --------------------------------------------
    A("## Read next")
    A("")
    A("`2°` rows ranked by relevance tier. Reading a PDF and adding locators is")
    A("what promotes a row to `1°` and makes it citable in a design argument.")
    A("")
    queue = [r for r in rows if _cell(r, "provenance") != "1"]
    queue.sort(key=lambda r: int(_cell(r, "relevance_tier", "9") or 9))
    if queue:
        A("| Tier | Paper | Why it matters |")
        A("|---|---|---|")
        for r in queue:
            A(
                f"| {_cell(r, 'relevance_tier')} | [{r['id']}](papers/{r['id']}.md) "
                f"| {_cell(r, 'headline', '—')} |"
            )
    else:
        A("*(queue empty — every row is `1°`)*")
    A("")

    # --- Section 3: the full table ---------------------------------------
    A("## All rows")
    A("")
    A("| ID | Venue | Obj | Family | Obs | Endpoint | Ident | Prov | Status |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: int(_cell(r, "relevance_tier", "9") or 9)):
        prov = "1°" if _cell(r, "provenance") == "1" else "2°"
        A(
            f"| [{r['id']}](papers/{r['id']}.md) | {_cell(r, 'venue', '—')} "
            f"| {_cell(r, 'object', '—')} | {_cell(r, 'model_family', '—')} "
            f"| {_cell(r, 'observability', '—')} | {_cell(r, 'endpoint', '—')} "
            f"| {_cell(r, 'identifiability_contact', '—')} | {prov} "
            f"| {_cell(r, 'status', '—')} |"
        )
    A("")

    # --- Section 4: the measurement --------------------------------------
    A("## What the pile looks like")
    A("")
    fams: dict[str, int] = {}
    obs: dict[str, int] = {}
    for r in rows:
        fams[_cell(r, "model_family", "—")] = fams.get(_cell(r, "model_family", "—"), 0) + 1
        obs[_cell(r, "observability", "—")] = obs.get(_cell(r, "observability", "—"), 0) + 1
    A("- **Model families:** " + ", ".join(f"`{k}` {v}" for k, v in sorted(fams.items())))
    A("- **Observability:** " + ", ".join(f"`{k}` {v}" for k, v in sorted(obs.items())))
    ident = sum(1 for r in rows if _cell(r, "identifiability_contact") == "yes")
    A(f"- **Bearing on identifiability (PROJECT.md §5):** {ident}")
    A("")
    A(f"*Generated {dt.date.today().isoformat()} by `research/lit_table.py`.*")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    rows = load()
    if not rows:
        print("no papers found in literature/papers/", file=sys.stderr)
        return 1

    rendered = render(rows)

    if args.stdout:
        print(rendered)
        return 0

    if args.write:
        INDEX.write_text(rendered)
        print(f"wrote {INDEX.relative_to(REPO)} ({len(rows)} rows)")
        return 0

    # --check: compare ignoring the generated-on date line, which changes daily
    if not INDEX.exists():
        print(f"FAIL {INDEX.relative_to(REPO)} does not exist; run --write")
        return 1

    def strip_date(text: str) -> str:
        return "\n".join(ln for ln in text.splitlines() if not ln.startswith("*Generated "))

    if strip_date(INDEX.read_text()) != strip_date(rendered):
        print(f"FAIL {INDEX.relative_to(REPO)} is stale; run --write")
        return 1

    print(f"OK   literature index is current ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
