#!/bin/bash
# Run Lilith Tests
# Activates venv and runs the test suite

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Run setup first:"
    echo "   ./scripts/setup.sh"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "🧪 Running Lilith Tests..."
echo ""

# Default to running all tests, but allow specific test file as argument
if [ $# -eq 0 ]; then
    python -m pytest tests/ -v --tb=short
else
    python -m pytest "$@" -v --tb=short
fi
