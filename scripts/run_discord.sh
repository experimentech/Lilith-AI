#!/bin/bash
# Run Lilith Discord Bot
# Activates venv and starts the bot

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

# Check for .env file
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Create one from the template:"
    echo "   cp .env.example .env"
    echo "   Then edit .env with your Discord bot token"
    exit 1
fi

# Check for Discord token
if grep -q "your_discord_bot_token_here" .env 2>/dev/null; then
    echo "❌ Discord token not configured. Edit .env and add your bot token."
    exit 1
fi

# Check for discord.py
if ! python -c "import discord" 2>/dev/null; then
    echo "❌ discord.py not installed. Install it with:"
    echo "   pip install discord.py python-dotenv"
    exit 1
fi

# Show usage if --help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: run_discord.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --console    Run with interactive console"
    echo "  -t, --timeout N  Session timeout in minutes (default: 30)"
    echo "  -r, --retention N  User data retention in days (default: 7)"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Console mode provides commands like 'stats', 'users', 'cleanup', 'quit'"
    exit 0
fi

echo "🚀 Starting Lilith Discord Bot..."
echo ""

# Run the bot with any passed arguments
python discord_bot.py "$@"
