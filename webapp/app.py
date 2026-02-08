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

from flask import Flask, render_template, request, jsonify, Response

from optimizer import OptimizationParams, AdvancedOptimizationParams, run_optimization, run_advanced_optimization
from charts import generate_all_charts
from charts_advanced import generate_advanced_charts
from frontier import FrontierParams, run_frontier_analysis, export_frontier_csv
from charts_frontier import generate_frontier_charts

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


@app.route("/api/optimize-advanced", methods=["POST"])
def api_optimize_advanced():
    """
    Run advanced CVaR optimization with scenarios, constraints, and backtest.
    """
    try:
        if check_rate_limit(request.remote_addr):
            return jsonify({"error": "요청이 너무 많습니다. 잠시 후 다시 시도하세요."}), 429

        data = request.get_json()
        if data is None:
            return jsonify({"error": "유효한 JSON 요청이 필요합니다."}), 400

        # Validate tickers
        tickers = [t.strip().upper() for t in data.get("tickers", []) if t.strip()]
        if len(tickers) < 2:
            return jsonify({"error": "종목을 2개 이상 입력해주세요."}), 400
        if len(tickers) > 30:
            return jsonify({"error": "최대 30개 종목까지 가능합니다."}), 400

        # Validate dates
        start_date = data.get("start_date", "2022-01-01")
        end_date = data.get("end_date", date.today().isoformat())
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
            if start_dt >= end_dt:
                return jsonify({"error": "시작일은 종료일보다 이전이어야 합니다."}), 400
        except ValueError:
            return jsonify({"error": "날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요."}), 400

        # Numeric parameters with defaults
        try:
            confidence_level = float(data.get("confidence_level", 0.95))
            risk_aversion = float(data.get("risk_aversion", 0.5))
            min_weight = float(data.get("min_weight", 0.0))
            max_weight = float(data.get("max_weight", 1.0))
        except (TypeError, ValueError):
            return jsonify({"error": "숫자 매개변수가 올바르지 않습니다."}), 400

        if not (0.90 <= confidence_level <= 0.99):
            return jsonify({"error": "신뢰수준은 0.90~0.99 범위여야 합니다."}), 400

        # Scenario settings
        return_type = data.get("return_type", "simple")
        if return_type not in ("simple", "log"):
            return jsonify({"error": "수익률 유형은 simple 또는 log이어야 합니다."}), 400

        scenario_method = data.get("scenario_method", "historical")
        if scenario_method not in ("historical", "kde", "gaussian"):
            return jsonify({"error": "시나리오 방법은 historical, kde, gaussian 중 하나여야 합니다."}), 400

        num_scenarios = int(data.get("num_scenarios", 0))
        kde_bandwidth = float(data.get("kde_bandwidth", 0.5))
        kde_kernel = data.get("kde_kernel", "gaussian")

        # Per-asset bounds
        asset_bounds = data.get("asset_bounds", [])

        # Cash
        cash_min = float(data.get("cash_min", 0.0))
        cash_max = float(data.get("cash_max", 0.0))

        # Leverage
        leverage_target = float(data.get("leverage_target", 1.0))

        # Turnover
        turnover_target = data.get("turnover_target")
        if turnover_target is not None:
            turnover_target = float(turnover_target)
        current_weights = data.get("current_weights")

        # CVaR limit
        cvar_limit = data.get("cvar_limit")
        if cvar_limit is not None:
            cvar_limit = float(cvar_limit)

        # Cardinality
        max_assets = data.get("max_assets")
        if max_assets is not None:
            max_assets = int(max_assets)

        # Backtest
        test_split_date = data.get("test_split_date")
        benchmark_portfolios = data.get("benchmark_portfolios")

        params = AdvancedOptimizationParams(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            confidence_level=confidence_level,
            risk_aversion=risk_aversion,
            min_weight=min_weight,
            max_weight=max_weight,
            return_type=return_type,
            scenario_method=scenario_method,
            num_scenarios=num_scenarios,
            kde_bandwidth=kde_bandwidth,
            kde_kernel=kde_kernel,
            asset_bounds=asset_bounds,
            cash_min=cash_min,
            cash_max=cash_max,
            leverage_target=leverage_target,
            turnover_target=turnover_target,
            current_weights=current_weights,
            cvar_limit=cvar_limit,
            max_assets=max_assets,
            test_split_date=test_split_date,
            benchmark_portfolios=benchmark_portfolios,
        )

        result = run_advanced_optimization(params)
        charts = generate_advanced_charts(result)
        bt = result.backtest

        summary = {
            "weights": result.weights,
            "cash_weight": result.cash_weight,
            "expected_return": result.expected_return,
            "cvar": result.cvar,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "data_points": len(result.returns_df),
            "date_range": f"{result.prices_df.index[0].date()} ~ {result.prices_df.index[-1].date()}",
            "scenario_method": params.scenario_method,
            "num_scenarios": result.scenarios.shape[0],
        }

        # Train/test metrics
        if bt.train_sharpe is not None:
            summary["train_sharpe"] = bt.train_sharpe
            summary["train_sortino"] = bt.train_sortino
            summary["test_sharpe"] = bt.test_sharpe
            summary["test_sortino"] = bt.test_sortino

        # Benchmark metrics
        if bt.benchmark_metrics:
            summary["benchmark_metrics"] = bt.benchmark_metrics

        return jsonify({"summary": summary, "charts": charts})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        logger.exception("Advanced optimization failed")
        return jsonify({"error": "내부 오류가 발생했습니다. 다른 매개변수로 다시 시도해주세요."}), 500


@app.route("/api/frontier", methods=["POST"])
def api_frontier():
    """Run efficient frontier analysis with logarithmic lambda sweep."""
    try:
        if check_rate_limit(request.remote_addr):
            return jsonify({"error": "요청이 너무 많습니다. 잠시 후 다시 시도하세요."}), 429

        data = request.get_json()
        if data is None:
            return jsonify({"error": "유효한 JSON 요청이 필요합니다."}), 400

        tickers = [t.strip().upper() for t in data.get("tickers", []) if t.strip()]
        if len(tickers) < 2:
            return jsonify({"error": "종목을 2개 이상 입력해주세요."}), 400
        if len(tickers) > 30:
            return jsonify({"error": "최대 30개 종목까지 가능합니다."}), 400

        start_date = data.get("start_date", "2022-01-01")
        end_date = data.get("end_date", date.today().isoformat())
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
            if start_dt >= end_dt:
                return jsonify({"error": "시작일은 종료일보다 이전이어야 합니다."}), 400
        except ValueError:
            return jsonify({"error": "날짜 형식이 올바르지 않습니다."}), 400

        try:
            confidence_level = float(data.get("confidence_level", 0.95))
            min_weight = float(data.get("min_weight", 0.0))
            max_weight = float(data.get("max_weight", 1.0))
            min_exp = float(data.get("min_exp", -3.0))
            max_exp = float(data.get("max_exp", 1.0))
            n_steps = int(data.get("n_steps", 50))
        except (TypeError, ValueError):
            return jsonify({"error": "숫자 매개변수가 올바르지 않습니다."}), 400

        if not (0.90 <= confidence_level <= 0.99):
            return jsonify({"error": "신뢰수준은 0.90~0.99 범위여야 합니다."}), 400
        if n_steps < 5 or n_steps > 200:
            return jsonify({"error": "프론티어 단계 수는 5~200 범위여야 합니다."}), 400

        custom_portfolio = data.get("custom_portfolio", {})

        params = FrontierParams(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            confidence_level=confidence_level,
            min_weight=min_weight,
            max_weight=max_weight,
            min_exp=min_exp,
            max_exp=max_exp,
            n_steps=n_steps,
            return_type=data.get("return_type", "simple"),
            scenario_method=data.get("scenario_method", "historical"),
            num_scenarios=int(data.get("num_scenarios", 0)),
            kde_bandwidth=float(data.get("kde_bandwidth", 0.5)),
            kde_kernel=data.get("kde_kernel", "gaussian"),
            custom_portfolio=custom_portfolio,
        )

        result = run_frontier_analysis(params)
        charts = generate_frontier_charts(result)

        # Tangent portfolio info
        tang = result.points[result.tangent_idx]
        tangent_info = {
            "lambda": tang.lambda_val,
            "expected_return": tang.expected_return,
            "cvar": tang.cvar,
            "volatility": tang.volatility,
            "weights": tang.weights,
        }

        summary = {
            "n_points": len(result.points),
            "tangent": tangent_info,
            "tickers": result.tickers,
        }
        if result.custom_eval:
            summary["custom_eval"] = {
                "expected_return": result.custom_eval.expected_return,
                "cvar": result.custom_eval.cvar,
                "volatility": result.custom_eval.volatility,
                "sharpe": result.custom_eval.sharpe,
                "weights": result.custom_eval.weights,
            }

        # Include frontier data for export
        frontier_data = [{
            "lambda": p.lambda_val,
            "expected_return": p.expected_return,
            "cvar": p.cvar,
            "volatility": p.volatility,
            "weights": p.weights,
        } for p in result.points]

        return jsonify({
            "summary": summary,
            "charts": charts,
            "frontier_data": frontier_data,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        logger.exception("Frontier analysis failed")
        return jsonify({"error": "내부 오류가 발생했습니다. 다른 매개변수로 다시 시도해주세요."}), 500


@app.route("/api/frontier/export", methods=["POST"])
def api_frontier_export():
    """Export frontier data as CSV."""
    try:
        if check_rate_limit(request.remote_addr):
            return jsonify({"error": "요청이 너무 많습니다."}), 429

        data = request.get_json()
        if data is None:
            return jsonify({"error": "유효한 JSON 요청이 필요합니다."}), 400

        tickers = [t.strip().upper() for t in data.get("tickers", []) if t.strip()]
        if len(tickers) < 2:
            return jsonify({"error": "종목을 2개 이상 입력해주세요."}), 400

        params = FrontierParams(
            tickers=tickers,
            start_date=data.get("start_date", "2022-01-01"),
            end_date=data.get("end_date", date.today().isoformat()),
            confidence_level=float(data.get("confidence_level", 0.95)),
            min_weight=float(data.get("min_weight", 0.0)),
            max_weight=float(data.get("max_weight", 1.0)),
            min_exp=float(data.get("min_exp", -3.0)),
            max_exp=float(data.get("max_exp", 1.0)),
            n_steps=int(data.get("n_steps", 50)),
            return_type=data.get("return_type", "simple"),
            scenario_method=data.get("scenario_method", "historical"),
            num_scenarios=int(data.get("num_scenarios", 0)),
            kde_bandwidth=float(data.get("kde_bandwidth", 0.5)),
            kde_kernel=data.get("kde_kernel", "gaussian"),
        )

        result = run_frontier_analysis(params)
        csv_str = export_frontier_csv(result.points, result.tickers)

        return Response(
            csv_str,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=frontier.csv"},
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        logger.exception("Frontier export failed")
        return jsonify({"error": "내부 오류가 발생했습니다."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
