#!/bin/bash
# Lilith AI Setup Script
# Creates virtual environment and installs all dependencies

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "  Lilith AI Setup"
echo "=================================================="
echo ""

cd "$PROJECT_ROOT"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Found Python $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.10+ is required. Please upgrade Python."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "   ✓ Created .venv/"
else
    echo ""
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install core dependencies
echo ""
echo "📥 Installing core dependencies..."
pip install -e . --quiet 2>/dev/null || pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "   Installing from pyproject.toml..."
    pip install numpy scipy scikit-learn --quiet
    pip install pytest pytest-cov --quiet
}

# Check for optional Discord dependencies
echo ""
read -p "🎮 Install Discord bot dependencies? (y/N): " install_discord
if [[ "$install_discord" =~ ^[Yy]$ ]]; then
    echo "   Installing discord.py and python-dotenv..."
    pip install discord.py python-dotenv --quiet
    echo "   ✓ Discord dependencies installed"
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        echo ""
        echo "📝 Creating .env file from template..."
        cp .env.example .env
        echo "   ✓ Created .env (edit this with your Discord token)"
    fi
fi

# Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/base
mkdir -p data/users
echo "   ✓ Created data/base/ and data/users/"

# Summary
echo ""
echo "=================================================="
echo "  ✅ Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  python lilith_cli.py"
echo ""
if [[ "$install_discord" =~ ^[Yy]$ ]]; then
    echo "To run the Discord bot:"
    echo "  1. Edit .env with your Discord bot token"
    echo "  2. python discord_bot.py"
    echo ""
fi
echo "To run tests:"
echo "  python -m pytest tests/ -v"
echo ""
