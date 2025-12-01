#!/bin/bash
# Installation script for Lilith AI

set -e  # Exit on error

echo "=========================================="
echo "Installing Lilith AI Dependencies"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Core dependencies installed"
echo ""

# Download NLTK data for WordNet
echo "Downloading NLTK WordNet data..."
python3 -c "
import nltk
import sys

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print('✅ WordNet data downloaded')
except Exception as e:
    print(f'⚠️  Warning: Could not download WordNet data: {e}')
    print('   WordNet features will be unavailable')
    sys.exit(0)  # Don't fail installation
"
echo ""

# Install Discord bot dependencies (optional)
read -p "Install Discord bot dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Discord dependencies..."
    pip install discord.py python-dotenv --quiet
    echo "✅ Discord dependencies installed"
    echo ""
    echo "⚠️  Don't forget to create a .env file with your DISCORD_TOKEN"
else
    echo "⏭️  Skipping Discord dependencies"
fi
echo ""

echo "=========================================="
echo "Installation Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run tests: python tests/test_conversation.py"
echo "  3. Run CLI: python lilith_cli.py"
echo ""
echo "For Discord bot:"
echo "  1. Create .env file with DISCORD_TOKEN=your_token"
echo "  2. Run: python discord_bot.py"
echo ""
