# Makefile for optimize-framework

.PHONY: help install install-dev test lint format docs clean

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  docs         Build documentation"
	@echo "  clean        Clean build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=optimizer_framework

lint:
	flake8 optimizer_framework tests
	mypy optimizer_framework
	black --check optimizer_framework tests
	isort --check-only optimizer_framework tests

format:
	black optimizer_framework tests scripts
	isort optimizer_framework tests scripts

docs:
	nbdev_docs

clean:
	rm -rf build dist *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
