.PHONY: install test test-integration test-security test-all lint format typecheck run clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/unit/ tests/security/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-security:
	pytest tests/security/ -v -m security

test-all:
	pytest tests/ -v

lint:
	ruff check src/ tests/ app.py
	ruff format --check src/ tests/ app.py

format:
	ruff check --fix src/ tests/ app.py
	ruff format src/ tests/ app.py

typecheck:
	mypy src/

secret-scan:
	detect-secrets scan --all-files --exclude-files '\.env\.example'

run:
	python app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
