UV ?= uv

.PHONY: install lint format format-check typecheck security test build check run lock

install:
	$(UV) sync --locked

format:
	$(UV) run --locked ruff format kairos_aggregator tests

format-check:
	$(UV) run --locked ruff format --check kairos_aggregator tests

lint:
	$(UV) run --locked ruff check kairos_aggregator tests

typecheck:
	$(UV) run --locked mypy kairos_aggregator

security:
	$(UV) run --locked bandit -q -r kairos_aggregator -x tests

test:
	$(UV) run --locked pytest -q --tb=short

build:
	$(UV) build --no-sources

check: lint format-check typecheck security test build

run:
	$(UV) run --locked python -m kairos_aggregator

lock:
	$(UV) lock
