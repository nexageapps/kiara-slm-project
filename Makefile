.PHONY: help install install-dev test lint format clean train evaluate docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install       - Install package and dependencies"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make train         - Run training"
	@echo "  make evaluate      - Run evaluation"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

docker-build:
	docker build -t kiara-slm:latest .

docker-run:
	docker run --gpus all -v $(PWD)/data:/app/data -v $(PWD)/checkpoints:/app/checkpoints kiara-slm:latest
