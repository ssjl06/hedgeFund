#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -q -r requirements.txt 2>/dev/null || true

if [[ "$1" == "--prod" ]]; then
    echo "Starting production server on http://0.0.0.0:8000"
    gunicorn -c deploy/gunicorn.conf.py app:app
else
    echo "Starting development server on http://0.0.0.0:5000"
    python app.py
fi
