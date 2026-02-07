"""
CVaR Portfolio Optimization Web Server

Flask application wrapping the Mean-CVaR pipeline from
NVIDIA's cvar_basic.ipynb notebook into an interactive web UI.
"""

import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import date, timedelta

from flask import Flask, render_template, request, jsonify

from optimizer import OptimizationParams, run_optimization
from charts import generate_all_charts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading Flask...")
logger.info("Loading optimizer...")
logger.info("Loading charts...")
logger.info("All modules loaded.")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32).hex())

# Trust X-Forwarded-* headers from Nginx reverse proxy
try:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
except ImportError:
    pass

# Simple in-memory rate limiting
_rate_limit = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10     # requests per window


def check_rate_limit(ip: str) -> bool:
    """Return True if rate limit exceeded."""
    now = time.time()
    _rate_limit[ip] = [t for t in _rate_limit[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit[ip]) >= RATE_LIMIT_MAX:
        return True
    _rate_limit[ip].append(now)
    return False


@app.route("/")
def index():
    """Serve the main single-page application."""
    today = date.today().isoformat()
    two_years_ago = (date.today() - timedelta(days=730)).isoformat()
    return render_template("index.html", today=today, default_start=two_years_ago)


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """
    Run the CVaR portfolio optimization and return results + charts.

    Expected JSON body:
    {
      "tickers": ["AAPL", "MSFT", ...],
      "start_date": "2022-01-01",
      "end_date": "2024-12-31",
      "confidence_level": 0.95,
      "risk_aversion": 0.5,
      "min_weight": 0.0,
      "max_weight": 1.0
    }
    """
    try:
        # Check rate limit
        if check_rate_limit(request.remote_addr):
            return jsonify({"error": "Too many requests. Please wait before trying again."}), 429

        # Get and validate JSON
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json."}), 400

        # Validate tickers
        tickers = [t.strip().upper() for t in data.get("tickers", []) if t.strip()]
        if len(tickers) < 2:
            return jsonify({"error": "At least 2 tickers are required."}), 400
        if len(tickers) > 30:
            return jsonify({"error": "Maximum 30 tickers allowed."}), 400

        # Validate dates
        start_date = data.get("start_date", "2022-01-01")
        end_date = data.get("end_date", date.today().isoformat())
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
            if start_dt >= end_dt:
                return jsonify({"error": "Start date must be before end date."}), 400
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        # Validate numeric parameters
        try:
            confidence_level = float(data.get("confidence_level", 0.95))
            risk_aversion = float(data.get("risk_aversion", 0.5))
            min_weight = float(data.get("min_weight", 0.0))
            max_weight = float(data.get("max_weight", 1.0))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid numeric parameter."}), 400

        if not (0.90 <= confidence_level <= 0.99):
            return jsonify({"error": "Confidence level must be between 0.90 and 0.99."}), 400
        if not (0.0 <= risk_aversion <= 1.0):
            return jsonify({"error": "Risk aversion must be between 0.0 and 1.0."}), 400
        if min_weight > max_weight:
            return jsonify({"error": "Minimum weight must not exceed maximum weight."}), 400
        if not (-1.0 <= min_weight <= 1.0) or not (-1.0 <= max_weight <= 1.0):
            return jsonify({"error": "Weight bounds must be between -1.0 and 1.0."}), 400

        # Construct parameters with validated values
        params = OptimizationParams(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            confidence_level=confidence_level,
            risk_aversion=risk_aversion,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        result = run_optimization(params)
        charts = generate_all_charts(result)

        summary = {
            "weights": result.weights,
            "expected_return": result.expected_return,
            "cvar": result.cvar,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "data_points": len(result.returns_df),
            "date_range": f"{result.prices_df.index[0].date()} ~ {result.prices_df.index[-1].date()}",
        }

        return jsonify({"summary": summary, "charts": charts})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        logger.exception("Optimization failed")
        return jsonify({"error": "An internal error occurred. Please try again with different parameters."}), 500


@app.route("/api/presets", methods=["GET"])
def api_presets():
    """Preset ticker groups for quick selection."""
    presets = {
        "US Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "S&P Sector ETFs": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"],
        "Global Diversified": ["SPY", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC"],
        "FAANG + Semiconductor": ["AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA", "AMD", "AVGO", "TSM"],
        "Korean ADR": ["005930.KS", "000660.KS", "035420.KS", "035720.KS", "051910.KS", "006400.KS"],
    }
    return jsonify(presets)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
