# Task orchestration. Every target is runnable via `uv run`, so a bare checkout
# needs only `make setup`.
.PHONY: setup setup-gpu lint fmt test audit lit-check all

setup:
	uv sync --group dev

# CUDA JAX from the lock. Verified 2026-09-23: once installed, a bare `uv run`
# KEEPS it and GPU becomes the default backend for everything, tests included
# (51 tests: 109 s on GPU vs ~15 s on CPU). Force CPU per command with
# JAX_PLATFORMS=cpu; `make setup` (exact sync) removes the extra entirely.
setup-gpu:
	uv sync --group dev --extra gpu

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -m "not slow"

# Hard-fails on a broken RESEARCH_LOG line-link or a stale generated file.
audit:
	uv run python tools/doc_audit.py

# Fails if literature/PAPERS.md is stale relative to literature/papers/*.md.
lit-check:
	uv run python research/lit_table.py --check

all: lint test audit lit-check

# 2026-10-17: the .tex comments carry log line numbers and "our program" wording that
# would deanonymise a source upload. Writes paper/submission/ with comments stripped.
paper-anon:
	mkdir -p paper/submission/figures
	sed -E 's/(^|[^\\])%.*$$/\1/' paper/main.tex > paper/submission/main.tex
	cp paper/numbers.tex paper/tmlr.sty paper/tmlr.bst paper/refs.bib paper/figures/*.pdf paper/submission/ 2>/dev/null; mv paper/submission/fig_*.pdf paper/submission/figures/ 2>/dev/null; true

# 2026-10-19: arXiv source (non-anonymous, `preprint` option, comments stripped because they carry
# log line numbers). Set the author block: make paper-arxiv ARXIV_AUTHOR='\name A. Name \email a@b.c \\ \addr Institution'
# The bibliography is inline (no .bib); the zip is paper/arxiv.zip. Do NOT upload paper/submission/ (anonymised).
ARXIV_AUTHOR ?= \name AUTHOR NAME \email AUTHOR@EMAIL \\ \addr AFFILIATION
paper-arxiv:
	mkdir -p paper/arxiv/figures
	sed -E 's/(^|[^\\])%.*$$/\1/' paper/main.tex \
	| sed -E 's/\\usepackage\{tmlr\}/\\usepackage[preprint]{tmlr}/' \
	| ARXIV_AUTHOR='$(ARXIV_AUTHOR)' awk '/^\\author\{/ {print "\\author{" ENVIRON["ARXIV_AUTHOR"] "}"; next} {print}' > paper/arxiv/main.tex
	cp paper/numbers.tex paper/tmlr.sty paper/arxiv/
	cp paper/figures/*.pdf paper/arxiv/figures/
	cd paper/arxiv && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null
	cd paper/arxiv && rm -f ../arxiv.zip && zip -q ../arxiv.zip main.tex numbers.tex tmlr.sty figures/*.pdf
