#!/usr/bin/env python3
"""Documentation-integrity audit for the adaptive-therapy research record.

The repo's memory is a past/present/future split: an append-only RESEARCH_LOG.md
(past), regenerable VERDICTS/CLOSED/LAWS indices (present), and roadmap docs
(future). That design only works while a handful of invariants hold, and every
one of them is otherwise maintained by hand -- i.e. by discipline.

This script turns those invariants into a check that fails:

  1. LINK INTEGRITY   every `RESEARCH_LOG.md#Lnnnn` resolves to a real line.
  2. INDEX COVERAGE   every log entry is reachable from some index. CLAUDE.md:
                      "an entry without an index update is invisible to future
                      sessions."
  3. ENTRY BUDGET     entries stay under the 40-line cap. Overflow prose, not
                      entry count, is what pushes a log past 5k lines.
  4. APPEND-ONLY      old entries are never edited in place. This is what keeps
                      the `#Lnnnn` links stable; violating it silently
                      invalidates every link below the edit.
  5. FORWARD-DOC      past-tense content rotting in forward-looking docs.
     STALENESS

Usage:
    uv run python tools/doc_audit.py              # summary
    uv run python tools/doc_audit.py -v           # every finding
    uv run python tools/doc_audit.py --max-lines 40

Exit code 1 only if a HARD invariant is broken: a link that no longer resolves.
Everything else reports as a WARN and still exits 0 -- it is a backlog to burn
down, not a build break.

Append-only is deliberately soft: git history is immutable, so a hard fail there
would stay red forever and train everyone to ignore the tool; its actual damage
(shifted line links) is caught hard by the link check.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "outputs", ".ruff_cache"}

LOG_NAME = "RESEARCH_LOG.md"
ENTRY_RE = re.compile(r"^## \[")
# e.g. [L134](RESEARCH_LOG.md#L134) or [rl L12](adaptive_therapy/RESEARCH_LOG.md#L12)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]*RESEARCH_LOG\.md)#L(\d+)\)")
FORWARD_DOCS = ("ROADMAP.md", "PREREG.md", "NEXT_STEPS.md", "FUTURE_WORK.md")
STALE_MARKERS = ("~~", "✅", "DONE", "SUPERSEDED")


def _colour(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


RED = lambda s: _colour("31", s)  # noqa: E731
YEL = lambda s: _colour("33", s)  # noqa: E731
GRN = lambda s: _colour("32", s)  # noqa: E731


@dataclass
class Report:
    hard: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.hard.append(msg)

    def note(self, msg: str) -> None:
        self.warn.append(msg)


def md_files() -> list[Path]:
    out = []
    for p in REPO.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def log_files() -> list[Path]:
    return [p for p in md_files() if p.name == LOG_NAME]


def check_links(rep: Report) -> None:
    """HARD: every RESEARCH_LOG.md#Lnnnn target must exist."""
    line_counts: dict[Path, int] = {}
    for log in log_files():
        line_counts[log] = len(log.read_text().splitlines())

    for src in md_files():
        for lineno, line in enumerate(src.read_text().splitlines(), 1):
            for target, num in LINK_RE.findall(line):
                resolved = (src.parent / target).resolve()
                if resolved not in line_counts:
                    rep.fail(
                        f"{src.relative_to(REPO)}:{lineno} -> {target}#L{num} (no such log file)"
                    )
                elif int(num) > line_counts[resolved]:
                    rep.fail(
                        f"{src.relative_to(REPO)}:{lineno} -> {target}#L{num} "
                        f"(log has {line_counts[resolved]} lines)"
                    )


def check_coverage(rep: Report) -> None:
    """WARN: every log entry should be linked from at least one index."""
    linked: set[tuple[Path, int]] = set()
    for src in md_files():
        for line in src.read_text().splitlines():
            for target, num in LINK_RE.findall(line):
                linked.add(((src.parent / target).resolve(), int(num)))

    for log in log_files():
        for lineno, line in enumerate(log.read_text().splitlines(), 1):
            if ENTRY_RE.match(line) and (log, lineno) not in linked:
                rep.note(
                    f"{log.relative_to(REPO)}:{lineno} entry has no inbound "
                    f"index link: {line.strip()[:60]}"
                )


def check_entry_budget(rep: Report, max_lines: int) -> None:
    """WARN: entries over the hard cap needed an artifact, not more prose."""
    for log in log_files():
        lines = log.read_text().splitlines()
        starts = [i for i, ln in enumerate(lines) if ENTRY_RE.match(ln)]
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
            length = end - start
            if length > max_lines:
                rep.note(
                    f"{log.relative_to(REPO)}:{start + 1} entry is {length} "
                    f"lines (cap {max_lines}): {lines[start].strip()[:50]}"
                )


def check_append_only(rep: Report) -> None:
    """WARN: a commit that DELETED lines from a log rewrote history."""
    try:
        out = subprocess.run(
            ["git", "log", "--numstat", "--format=%H", "--", f"*{LOG_NAME}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if out.returncode != 0:
        return

    sha = ""
    for line in out.stdout.splitlines():
        if re.fullmatch(r"[0-9a-f]{40}", line.strip()):
            sha = line.strip()[:9]
            continue
        parts = line.split("\t")
        if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) > 0:
            rep.note(f"{sha} deleted {parts[1]} line(s) from {parts[2]}")


def check_forward_staleness(rep: Report) -> None:
    """WARN: past-tense content accreting in forward-looking docs."""
    for src in md_files():
        if src.name not in FORWARD_DOCS:
            continue
        for lineno, line in enumerate(src.read_text().splitlines(), 1):
            for marker in STALE_MARKERS:
                if marker in line:
                    rep.note(
                        f"{src.relative_to(REPO)}:{lineno} carries '{marker}' "
                        f"-- settled content belongs in the log/index"
                    )
                    break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--max-lines", type=int, default=40)
    args = ap.parse_args()

    rep = Report()
    check_links(rep)
    check_coverage(rep)
    check_entry_budget(rep, args.max_lines)
    check_append_only(rep)
    check_forward_staleness(rep)

    limit = None if args.verbose else 10

    if rep.hard:
        print(RED(f"FAIL  {len(rep.hard)} broken link(s)"))
        for msg in rep.hard[:limit]:
            print(f"  {msg}")
        if limit and len(rep.hard) > limit:
            print(f"  ... {len(rep.hard) - limit} more (-v for all)")

    if rep.warn:
        print(YEL(f"WARN  {len(rep.warn)} finding(s)"))
        for msg in rep.warn[:limit]:
            print(f"  {msg}")
        if limit and len(rep.warn) > limit:
            print(f"  ... {len(rep.warn) - limit} more (-v for all)")

    if not rep.hard and not rep.warn:
        print(GRN("OK    research record is consistent"))
    elif not rep.hard:
        print(GRN("OK    no hard failures"))

    return 1 if rep.hard else 0


if __name__ == "__main__":
    sys.exit(main())
