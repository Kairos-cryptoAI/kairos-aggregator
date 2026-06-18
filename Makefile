.PHONY: install lint test format run
install:
	pip install -e ".[dev]"
format:
	ruff format kairos_aggregator tests
lint:
	ruff check kairos_aggregator tests
test:
	pytest -q
run:
	python -m kairos_aggregator
