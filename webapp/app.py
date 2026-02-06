"""
CVaR Portfolio Optimization Web Server

Flask application wrapping the Mean-CVaR pipeline from
NVIDIA's cvar_basic.ipynb notebook into an interactive web UI.
"""

import traceback
from datetime import date, timedelta

from flask import Flask, render_template, request, jsonify

from optimizer import OptimizationParams, run_optimization
from charts import generate_all_charts

app = Flask(__name__)


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
        data = request.get_json(force=True)

        tickers = [t.strip().upper() for t in data.get("tickers", []) if t.strip()]
        if len(tickers) < 2:
            return jsonify({"error": "At least 2 tickers are required."}), 400

        params = OptimizationParams(
            tickers=tickers,
            start_date=data.get("start_date", "2022-01-01"),
            end_date=data.get("end_date", date.today().isoformat()),
            confidence_level=float(data.get("confidence_level", 0.95)),
            risk_aversion=float(data.get("risk_aversion", 0.5)),
            min_weight=float(data.get("min_weight", 0.0)),
            max_weight=float(data.get("max_weight", 1.0)),
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

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
    app.run(host="0.0.0.0", port=5000, debug=True)
