#!/bin/bash
# Archive the Lilith project, excluding large/unnecessary directories
#
# Usage: ./archive_project.sh
# Creates: ../lilith-YYYYMMDD-HHMMSS.tar.gz

set -e

# Get project directory name and parent directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
PARENT_DIR="$(dirname "$PROJECT_DIR")"

# Generate timestamp for unique filename
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE_NAME="${PROJECT_NAME}-${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${PARENT_DIR}/${ARCHIVE_NAME}"

echo "Creating archive of ${PROJECT_NAME}..."
echo "Archive: ${ARCHIVE_PATH}"
echo ""

# Create archive excluding common large/unnecessary directories
cd "$PARENT_DIR"

tar -czf "$ARCHIVE_NAME" \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='node_modules' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.egg-info' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='.DS_Store' \
    --exclude='*.sqlite' \
    --exclude='*.db' \
    --exclude='runs/*.db' \
    --exclude='*.log' \
    "$PROJECT_NAME"

# Get archive size
ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)

echo ""
echo "✅ Archive created successfully!"
echo "   Path: ${ARCHIVE_PATH}"
echo "   Size: ${ARCHIVE_SIZE}"
echo ""
echo "To extract: tar -xzf ${ARCHIVE_NAME}"
