#!/bin/bash
# Installation script for Lilith AI

set -e  # Exit on error

echo "=========================================="
echo "Installing Lilith AI Dependencies"
echo "=========================================="
echo ""

# Required Python version
required_major=3
required_minor=10

# Function to check if a Python version meets requirements
check_python_version() {
    local python_cmd="$1"
    
    if ! command -v "$python_cmd" &> /dev/null; then
        return 1
    fi
    
    local version_full=$("$python_cmd" --version 2>&1 | awk '{print $2}')
    local major=$(echo "$version_full" | cut -d. -f1)
    local minor=$(echo "$version_full" | cut -d. -f2)
    
    if [ "$major" -gt "$required_major" ]; then
        echo "$python_cmd:$version_full"
        return 0
    elif [ "$major" -eq "$required_major" ] && [ "$minor" -ge "$required_minor" ]; then
        echo "$python_cmd:$version_full"
        return 0
    fi
    
    return 1
}

# Check Python version - try multiple options
echo "Checking for Python ${required_major}.${required_minor}+..."

PYTHON_CMD=""
PYTHON_VERSION=""

# Try various Python commands in order of preference
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    result=$(check_python_version "$cmd" 2>/dev/null) || continue
    if [ -n "$result" ]; then
        PYTHON_CMD=$(echo "$result" | cut -d: -f1)
        PYTHON_VERSION=$(echo "$result" | cut -d: -f2)
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python ${required_major}.${required_minor} or higher is required"
    echo ""
    echo "   Available Python versions on this system:"
    for cmd in python3 python3.6 python3.7 python3.8 python3.9 python3.10 python3.11 python3.12 python3.13; do
        if command -v "$cmd" &> /dev/null; then
            ver=$("$cmd" --version 2>&1)
            echo "     - $cmd: $ver"
        fi
    done
    echo ""
    echo "   To install Python 3.10+:"
    echo "     Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "     Or use pyenv: pyenv install 3.10"
    exit 1
fi

echo "✅ Found $PYTHON_CMD ($PYTHON_VERSION)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    "$PYTHON_CMD" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "   Make sure venv module is installed:"
        echo "   sudo apt install python${required_major}.${required_minor}-venv"
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
