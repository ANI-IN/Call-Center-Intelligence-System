.PHONY: install download-data test lint format typecheck run eval eval-transcription eval-summary eval-qa eval-judge eval-correlation clean

install:
	pip install -e ".[dev]"
	pre-commit install

download-data:
	python scripts/download_dataset.py

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-security:
	pytest tests/security/ -v -m security

test-all:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

secret-scan:
	detect-secrets scan --all-files --exclude-files '\.env\.example'

run:
	python app.py

eval:
	python -m src.evaluation.run_eval --all

eval-transcription:
	python -m src.evaluation.run_eval --transcription

eval-summary:
	python -m src.evaluation.run_eval --summary

eval-qa:
	python -m src.evaluation.run_eval --qa

eval-judge:
	python -m src.evaluation.run_eval --judge

eval-correlation:
	python -m src.evaluation.run_eval --correlation

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
