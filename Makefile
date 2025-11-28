.PHONY: help install install-dev test lint format clean train predict monitor

# Default target
help:
	@echo "Gene Classifier ML - Available Commands"
	@echo "========================================"
	@echo "make install        - Install production dependencies"
	@echo "make install-dev    - Install development dependencies"
	@echo "make test           - Run tests with coverage"
	@echo "make lint           - Run linting checks"
	@echo "make format         - Format code with black and isort"
	@echo "make clean          - Clean temporary files"
	@echo "make train          - Train model"
	@echo "make predict        - Make predictions"
	@echo "make monitor        - Generate monitoring report"
	@echo "make setup-hooks    - Setup pre-commit hooks"
	@echo "make validate-data  - Validate training data"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v

# Code Quality
lint:
	black --check .
	isort --check-only .
	pylint *.py --disable=all --enable=E,F
	mypy *.py --ignore-missing-imports

format:
	black .
	isort .

# Security
security:
	safety check
	bandit -r . -f json

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .mypy_cache .coverage htmlcov dist build

# ML Operations
train:
	python train.py --config config.yaml

predict:
	python predict.py --test-file test_data.csv --output predictions.txt

monitor:
	@python -c "from model_monitor import ModelMonitor; m = ModelMonitor(); print(m.generate_monitoring_report())"

# Setup
setup-hooks:
	pre-commit install
	@echo "Pre-commit hooks installed successfully"

validate-data:
	@python -c "import pandas as pd; from data_validator import DataValidator; \
		data = pd.read_csv('DM_Project_24.csv'); \
		validator = DataValidator(); \
		result = validator.validate(data); \
		validator.print_report(result)"

# Docker (optional)
docker-build:
	docker build -t gene-classifier:latest .

docker-run:
	docker run -v $(PWD)/data:/app/data gene-classifier:latest

# DVC
dvc-setup:
	dvc init
	dvc add DM_Project_24.csv test_data.csv

# All-in-one setup
setup: install-dev setup-hooks
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"
