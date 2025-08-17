#!/bin/bash

# Complete code quality check script
# This script runs all quality checks: formatting, linting, and tests

set -e

echo "🚀 Running complete code quality checks..."

# Change to the project root
cd "$(dirname "$0")/.."

echo "📦 Installing dependencies..."
uv sync

echo ""
echo "=== 1. FORMATTING CHECKS ==="
./scripts/format.sh

echo ""
echo "=== 2. LINTING CHECKS ==="
./scripts/lint.sh

echo ""
echo "=== 3. RUNNING TESTS ==="
cd backend && uv run pytest tests/ -v

echo ""
echo "🎉 All quality checks passed!"