.PHONY: fix lint test install dev

install:
	uv sync

dev:
	uv sync --extra dev

fix:
	uv run ruff check --select I --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	pytest tests/
