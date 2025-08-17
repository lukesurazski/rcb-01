#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"

# Change to backend directory and start the server
cd backend

# Force native arm64 3.13 so it satisfies requires-python >=3.13
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/opt/python@3.13/bin/python3.13}"
export UV_PYTHON="$PYTHON_BIN"

# optional: nuke any stale x86 venv to avoid confusion
[ -d .venv ] && rm -rf .venv

uv run --python "$PYTHON_BIN" uvicorn app:app --reload --port 8000



