# Task orchestration. Every target is runnable via `uv run`, so a bare checkout
# needs only `make setup`.
.PHONY: setup lint fmt test audit lit-check all

setup:
	uv sync --group dev

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
