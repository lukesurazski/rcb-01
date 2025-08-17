#!/bin/bash

# Code formatting script for the RAG system
# This script runs Black, isort, and other formatting tools

set -e

echo "🎨 Running code formatting..."

# Change to the project root
cd "$(dirname "$0")/.."

echo "📦 Installing dependencies..."
uv sync

echo "🔧 Running isort (import sorting)..."
uv run isort backend/ main.py --check-only --diff || {
    echo "❌ Import sorting issues found. Fixing..."
    uv run isort backend/ main.py
    echo "✅ Import sorting fixed."
}

echo "🖤 Running Black (code formatting)..."
uv run black backend/ main.py --check --diff || {
    echo "❌ Formatting issues found. Fixing..."
    uv run black backend/ main.py
    echo "✅ Code formatting fixed."
}

echo "✅ All formatting complete!"