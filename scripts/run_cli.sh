#!/bin/bash
# Run Lilith CLI (Interactive Mode)
# Activates venv and starts the command-line interface

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

echo "🧠 Starting Lilith CLI..."
echo ""

# Run the CLI
python lilith_cli.py "$@"
