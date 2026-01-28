#!/bin/bash
# Run all checks: lint, format, and tests

set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Running all checks ==="
echo ""

echo "=== 1. Ruff Lint ==="
ruff check src/ tests/ examples/ --fix
echo ""

echo "=== 2. Black Format ==="
black src/ tests/ examples/ --line-length 100
echo ""

echo "=== 3. Ruff Format ==="
ruff format src/ tests/ examples/
echo ""

echo "=== 4. Run Tests ==="
pytest tests/ -v --cov=contexo --cov-report=term-missing
echo ""

echo "=== All checks passed! ==="
