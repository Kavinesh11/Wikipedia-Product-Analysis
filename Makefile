.PHONY: help install test lint format clean setup

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make setup      - Initial project setup"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit -v -m unit

test-property:
	pytest tests/property -v -m property

lint:
	flake8 src tests
	mypy src

format:
	black src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

setup:
	cp .env.example .env
	mkdir -p logs data
	@echo "Setup complete! Edit .env with your configuration."
