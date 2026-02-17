#!/bin/bash
# Run the CVaR Portfolio Optimization web app using NVIDIA's Python environment
VENV_PYTHON=/workspace/quantitative-portfolio-optimization/.venv/bin/python
cd "$(dirname "$0")/webapp"
exec $VENV_PYTHON app.py "$@"
