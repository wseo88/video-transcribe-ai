# Makefile for Video Transcribe AI development

.PHONY: help install-dev format lint type-check test test-cov clean all

# Default target
help:
	@echo "Available commands:"
	@echo "  install-dev   Install development dependencies"
	@echo "  format        Format code with black and ruff"
	@echo "  lint          Run ruff linter"
	@echo "  type-check    Run mypy type checker"
	@echo "  pre-commit    Setup pre-commit hooks"
	@echo "  clean         Clean up cache files"
	@echo "  all           Run format, lint and type-check"

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Format code
format:
	black .
	ruff check . --fix

# Run linter
lint:
	ruff check .

# Run type checker
type-check:
	mypy .

# Setup pre-commit hooks
pre-commit:
	pre-commit install

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

# Run all checks
all: format lint type-check test