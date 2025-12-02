#!/bin/bash
# Installation script for Lilith AI

set -e  # Exit on error

echo "=========================================="
echo "Installing Lilith AI Dependencies"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."

# Get Python version components
python_version_full=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo "$python_version_full" | cut -d. -f1)
python_minor=$(echo "$python_version_full" | cut -d. -f2)

required_major=3
required_minor=10

# Compare versions properly (as integers, not decimals)
version_ok=false
if [ "$python_major" -gt "$required_major" ]; then
    version_ok=true
elif [ "$python_major" -eq "$required_major" ] && [ "$python_minor" -ge "$required_minor" ]; then
    version_ok=true
fi

if [ "$version_ok" = false ]; then
    echo "❌ Error: Python ${required_major}.${required_minor} or higher is required (found $python_version_full)"
    echo "   On Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "   Or use pyenv to install a newer Python version"
    exit 1
fi
echo "✅ Python $python_version_full detected"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "   Make sure python3-venv is installed:"
        echo "   sudo apt install python${python_major}.${python_minor}-venv"
        exit 1
    fi
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

# Upgrade pip using python -m pip (more reliable than bare pip)
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet 2>/dev/null || python -m pip install --upgrade pip
echo "✅ pip upgraded"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
python -m pip install -r requirements.txt --quiet 2>/dev/null || python -m pip install -r requirements.txt
echo "✅ Core dependencies installed"
echo ""

# Download NLTK data for WordNet
echo "Downloading NLTK WordNet data..."
python -c "
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
    python -m pip install discord.py python-dotenv --quiet 2>/dev/null || python -m pip install discord.py python-dotenv
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
