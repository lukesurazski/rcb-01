#!/bin/bash

# Linting script for the RAG system
# This script runs flake8 and other linting tools

set -e

echo "🔍 Running code linting..."

# Change to the project root
cd "$(dirname "$0")/.."

echo "📦 Installing dependencies..."
uv sync

echo "📋 Running flake8 (linting)..."
uv run flake8 backend/ main.py --count --statistics

echo "🔧 Running isort check..."
uv run isort backend/ main.py --check-only --diff

echo "🖤 Running Black check..."
uv run black backend/ main.py --check --diff

echo "✅ All linting checks passed!"