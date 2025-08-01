# Makefile for optimize-framework

.PHONY: help install install-dev test lint format docs clean

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=atlas

lint:
	mypy src
	black --check src tests
	isort --check-only src tests

format:
	black src tests scripts
	isort src tests scripts

clean:
	rm -rf build dist *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
