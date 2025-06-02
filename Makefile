.PHONY: venv install dev test coverage lint format clean check

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Make venv 
venv:
	@test -d $(VENV_DIR) || (echo "Building venv environment..." && python3 -m venv $(VENV_DIR))

# Install and make venv 
install: venv
	$(PIP) install -U pip
	$(PIP) install -e .[dev]

# Ruff 
lint:
	$(VENV_DIR)/bin/ruff src/ tests/

# Format 
format:
	$(VENV_DIR)/bin/black src/ tests/

# Tests
tests:
	$(VENV_DIR)/bin/pytest tests/*.py 

# Coverage 
coverage:
	$(VENV_DIR)/bin/pytest --cov=lazyagent --cov-report=lcov tests/*.py

# Clean 
clean:
	rm -rf .pytest_cache htmlcov .coverage __pycache__ .ruff_cache coverage.lcov $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -r {} +

# Lint + format + test + coverage
check: lint format test coverage
