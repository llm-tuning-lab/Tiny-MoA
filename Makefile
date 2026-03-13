.PHONY: help test lint format type-check check clean

help:
	@echo "Tiny-MoA - Development Targets"
	@echo "=============================="
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              Check code style with ruff"
	@echo "  make format            Format code with ruff"
	@echo "  make check             Run lint"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Remove cache files and build artifacts"

lint:
	python -m ruff check src/ scripts/

format:
	python -m ruff format src/ scripts/

check: lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info
	@echo "Cleaned up cache files and build artifacts"
