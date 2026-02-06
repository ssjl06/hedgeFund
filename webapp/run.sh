#!/bin/bash
# ---------------------------------------------------
# CVaR Portfolio Optimization Web Server Launcher
# ---------------------------------------------------
# Usage:
#   ./run.sh              → development mode (port 5000)
#   ./run.sh --prod       → gunicorn production mode (port 8000)
# ---------------------------------------------------

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install dependencies if needed
pip install -q flask yfinance pandas numpy scipy plotly gunicorn 2>/dev/null || true

if [[ "$1" == "--prod" ]]; then
    echo "Starting production server on http://0.0.0.0:8000"
    gunicorn app:app --bind 0.0.0.0:8000 --workers 2 --timeout 120
else
    echo "Starting development server on http://0.0.0.0:5000"
    python app.py
fi
