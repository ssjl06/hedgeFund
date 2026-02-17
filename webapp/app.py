"""CVaR Portfolio Optimization Web Application.

Uses NVIDIA cufolio library to provide:
1. CVaR Basic Optimization
2. Efficient Frontier Analysis
"""
import sys
import os
import json
import logging
import traceback
import time

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify

# Use NVIDIA venv's cufolio (with cuopt GPU support)
from cufolio import cvar_optimizer, cvar_utils, utils
from cufolio.cvar_parameters import CvarParameters
from cufolio import backtest

# Detect GPU solver availability
import cvxpy as cp
try:
    _HAS_CUOPT = 'CUOPT' in cp.installed_solvers()
except Exception:
    _HAS_CUOPT = False

DEFAULT_SOLVER = 'CLARABEL'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'nvidia-cufolio', 'data', 'stock_data', 'sp500.csv')


def _get_solver_settings():
    """Return solver settings based on available hardware."""
    if _HAS_CUOPT:
        return {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP",
                "time_limit": 15, "optimality": 1e-4}
    return {"solver": cp.CLARABEL, "verbose": False,
            "tol_gap_abs": 1e-4, "tol_gap_rel": 1e-4, "tol_feas": 1e-4}


def _get_scenario_settings(num_scen=10000, fit_type='gaussian'):
    """Return scenario generation settings."""
    if fit_type == 'kde':
        device = 'GPU' if _HAS_CUOPT else 'CPU'
        return {
            'num_scen': num_scen,
            'fit_type': 'kde',
            'kde_settings': {'bandwidth': 0.01, 'kernel': 'gaussian', 'device': device},
            'verbose': False
        }
    elif fit_type == 'no_fit':
        return {'num_scen': num_scen, 'fit_type': 'no_fit', 'verbose': False}
    else:
        return {'num_scen': num_scen, 'fit_type': 'gaussian', 'verbose': False}


def _parse_weight_bounds(val, default):
    """Parse weight bounds from user input (float or dict-like string)."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        try:
            return float(val)
        except ValueError:
            pass
        try:
            return json.loads(val.replace("'", '"'))
        except Exception:
            return default
    if isinstance(val, dict):
        return val
    return default


def _build_portfolio_chart(portfolio):
    """Build Plotly chart for portfolio allocation."""
    weights = portfolio.weights
    tickers = portfolio.tickers
    cash = portfolio.cash

    indices = np.where(np.abs(weights) > 0.005)[0]
    labels = [tickers[i] for i in indices]
    vals = [weights[i] for i in indices]

    if abs(cash) > 0.001:
        labels.append('Cash')
        vals.append(cash)

    long_labels = [l for l, v in zip(labels, vals) if v > 0]
    long_vals = [v for v in vals if v > 0]
    short_labels = [l for l, v in zip(labels, vals) if v < 0]
    short_vals = [abs(v) for v in vals if v < 0]

    fig = make_subplots(rows=1, cols=2 if short_vals else 1,
                        specs=[[{"type": "pie"}] * (2 if short_vals else 1)],
                        subplot_titles=["롱 포지션"] + (["숏 포지션"] if short_vals else []))

    fig.add_trace(go.Pie(
        labels=long_labels, values=long_vals,
        textinfo='label+percent', hole=0.3,
        marker=dict(colors=[f'hsl({210 + i * 15}, 70%, {50 + i * 3}%)' for i in range(len(long_labels))])
    ), row=1, col=1)

    if short_vals:
        fig.add_trace(go.Pie(
            labels=short_labels, values=short_vals,
            textinfo='label+percent', hole=0.3,
            marker=dict(colors=[f'hsl({0 + i * 20}, 70%, 55%)' for i in range(len(short_labels))])
        ), row=1, col=2)

    fig.update_layout(title_text="포트폴리오 배분", height=400, showlegend=True)
    return fig


def _build_backtest_chart(backtester, cut_off_date=None):
    """Build Plotly chart for backtest results."""
    fig = go.Figure()

    portfolio_returns = backtester.portfolio_returns
    if portfolio_returns is not None and len(portfolio_returns) > 0:
        cum_returns = (1 + portfolio_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_returns.index, y=cum_returns.values,
            name='최적화 포트폴리오', line=dict(color='blue', width=2)
        ))

    if hasattr(backtester, 'benchmark_returns_dict') and backtester.benchmark_returns_dict:
        colors = ['green', 'orange', 'red', 'purple', 'brown']
        for idx, (name, bench_returns) in enumerate(backtester.benchmark_returns_dict.items()):
            cum = (1 + bench_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                name=name, line=dict(color=colors[idx % len(colors)], dash='dash')
            ))

    if cut_off_date:
        fig.add_vline(x=cut_off_date, line_dash="dash", line_color="gray",
                      annotation_text="학습/테스트 분할")

    fig.update_layout(
        title="백테스트: 누적 수익률",
        xaxis_title="날짜", yaxis_title="누적 수익률",
        height=450, template="plotly_white"
    )
    return fig


def _build_allocation_table(portfolio):
    """Build allocation data for table display."""
    weights = portfolio.weights
    tickers = portfolio.tickers
    rows = []
    for i in range(len(tickers)):
        if abs(weights[i]) > 0.001:
            rows.append({
                'ticker': tickers[i],
                'weight': round(float(weights[i]), 6),
                'pct': round(float(weights[i]) * 100, 2),
                'position': '롱' if weights[i] > 0 else '숏'
            })
    rows.sort(key=lambda x: -abs(x['weight']))
    return rows


@app.route('/')
def index():
    return render_template('index.html', has_gpu=_HAS_CUOPT)


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run CVaR portfolio optimization."""
    try:
        data = request.get_json()
        logger.info("Optimize request: %s", json.dumps(data, default=str))

        start_date = data.get('start_date', '2021-01-01')
        end_date = data.get('end_date', '2024-01-01')
        confidence = float(data.get('confidence', 0.95))
        risk_aversion = float(data.get('risk_aversion', 1.0))
        w_min = _parse_weight_bounds(data.get('w_min'), -0.3)
        w_max = _parse_weight_bounds(data.get('w_max'), 0.4)
        c_min = float(data.get('c_min', 0.0))
        c_max = float(data.get('c_max', 0.2))
        L_tar = float(data.get('L_tar', 1.6))
        T_tar_raw = data.get('T_tar')
        T_tar = float(T_tar_raw) if T_tar_raw not in (None, '', 'null') else None
        cardinality_raw = data.get('cardinality')
        cardinality = int(cardinality_raw) if cardinality_raw not in (None, '', 'null', 0) else None
        num_scen = int(data.get('num_scen', 10000))
        fit_type = data.get('fit_type', 'gaussian')
        return_type = data.get('return_type', 'LOG')

        # Test period for backtest
        test_start = data.get('test_start', '2023-09-01')
        test_end = data.get('test_end', '2024-07-01')

        t0 = time.time()

        # Ensure data exists
        if not os.path.exists(DATA_PATH):
            utils.download_data(DATA_PATH)

        regime_dict = {"name": "user_regime", "range": (start_date, end_date)}
        returns_compute_settings = {'return_type': return_type, 'freq': 1}
        returns_dict = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)

        scenario_settings = _get_scenario_settings(num_scen, fit_type)
        returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_settings)

        cvar_params = CvarParameters(
            w_min=w_min, w_max=w_max,
            c_min=c_min, c_max=c_max,
            L_tar=L_tar, T_tar=T_tar,
            cvar_limit=None,
            cardinality=cardinality,
            risk_aversion=risk_aversion,
            confidence=confidence
        )

        cvar_problem = cvar_optimizer.CVaR(
            returns_dict=returns_dict,
            cvar_params=cvar_params,
            api_settings={"api": "cvxpy"}
        )

        solver_settings = _get_solver_settings()
        result_row, optimal_portfolio = cvar_problem.solve_optimization_problem(
            solver_settings=solver_settings,
            print_results=False
        )

        elapsed = time.time() - t0

        # Build charts
        portfolio_chart = _build_portfolio_chart(optimal_portfolio)
        allocation_table = _build_allocation_table(optimal_portfolio)

        # Backtest
        backtest_chart_json = None
        backtest_metrics = None
        try:
            test_regime_dict = {"name": "test", "range": (test_start, test_end)}
            test_returns_dict = utils.calculate_returns(DATA_PATH, test_regime_dict, returns_compute_settings)

            backtester_obj = backtest.portfolio_backtester(
                optimal_portfolio, test_returns_dict, 0.0, "historical"
            )
            backtest_result, _ = backtester_obj.backtest_against_benchmarks(
                plot_returns=False, cut_off_date=end_date
            )

            backtest_fig = _build_backtest_chart(backtester_obj, cut_off_date=end_date)
            backtest_chart_json = json.loads(plotly.io.to_json(backtest_fig))

            if backtest_result is not None and len(backtest_result) > 0:
                row = backtest_result.iloc[0]
                backtest_metrics = {}
                for col in backtest_result.columns:
                    val = row[col]
                    if isinstance(val, (np.floating, float)):
                        backtest_metrics[col] = round(float(val), 6)
                    else:
                        backtest_metrics[col] = str(val)
        except Exception as e:
            logger.warning("Backtest failed: %s", e)

        response = {
            'success': True,
            'results': {
                'solver': str(result_row.get('solver', DEFAULT_SOLVER)),
                'solve_time': round(float(result_row.get('solve time', elapsed)), 4),
                'expected_return': round(float(result_row.get('return', 0)), 6),
                'cvar': round(float(result_row.get('CVaR', 0)), 6),
                'objective': round(float(result_row.get('obj', 0)), 6),
                'num_assets': int(np.sum(np.abs(optimal_portfolio.weights) > 0.001)),
                'cash': round(float(optimal_portfolio.cash), 4),
                'total_elapsed': round(elapsed, 2),
            },
            'allocation': allocation_table,
            'portfolio_chart': json.loads(plotly.io.to_json(portfolio_chart)),
            'backtest_chart': backtest_chart_json,
            'backtest_metrics': backtest_metrics,
        }
        return jsonify(response)

    except Exception as e:
        logger.error("Optimize error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/frontier', methods=['POST'])
def frontier():
    """Generate efficient frontier."""
    try:
        data = request.get_json()
        logger.info("Frontier request: %s", json.dumps(data, default=str))

        start_date = data.get('start_date', '2022-01-01')
        end_date = data.get('end_date', '2024-07-01')
        confidence = float(data.get('confidence', 0.95))
        w_min = float(data.get('w_min', 0.0))
        w_max = float(data.get('w_max', 1.0))
        c_min = float(data.get('c_min', 0.0))
        c_max = float(data.get('c_max', 0.0))
        L_tar = float(data.get('L_tar', 1.0))
        num_scen = int(data.get('num_scen', 10000))
        fit_type = data.get('fit_type', 'gaussian')
        return_type = data.get('return_type', 'LOG')
        ra_steps = int(data.get('ra_steps', 30))
        min_ra_exp = float(data.get('min_ra_exp', -3))
        max_ra_exp = float(data.get('max_ra_exp', 1))

        # Custom portfolios
        custom_portfolios_raw = data.get('custom_portfolios', {})

        t0 = time.time()

        if not os.path.exists(DATA_PATH):
            utils.download_data(DATA_PATH)

        regime_dict = {"name": "ef_regime", "range": (start_date, end_date)}
        returns_compute_settings = {'return_type': return_type, 'freq': 1}
        returns_dict = utils.calculate_returns(DATA_PATH, regime_dict, returns_compute_settings)

        scenario_settings = _get_scenario_settings(num_scen, fit_type)
        returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_settings)

        ef_params = CvarParameters(
            w_min=w_min, w_max=w_max,
            c_min=c_min, c_max=c_max,
            L_tar=L_tar, T_tar=None,
            cvar_limit=None, cardinality=None,
            risk_aversion=1, confidence=confidence
        )

        solver_settings = _get_solver_settings()

        # Parse custom portfolios
        custom_dict = None
        if custom_portfolios_raw:
            custom_dict = {}
            for name, pdata in custom_portfolios_raw.items():
                weights = pdata.get('weights', {})
                cash = float(pdata.get('cash', 0.0))
                custom_dict[name] = (weights, cash)

        results_df, fig_mpl, ax = cvar_utils.create_efficient_frontier(
            returns_dict,
            ef_params,
            solver_settings,
            custom_portfolios_dict=custom_dict,
            ra_num=ra_steps,
            min_risk_aversion=min_ra_exp,
            max_risk_aversion=max_ra_exp,
            save_path=None,
            show_discretized_portfolios=False,
            print_portfolio_results=False,
            show_plot=False
        )
        import matplotlib
        matplotlib.pyplot.close('all')

        elapsed = time.time() - t0

        # Build Plotly frontier chart
        fig = go.Figure()
        if results_df is not None and len(results_df) > 0:
            fig.add_trace(go.Scatter(
                x=results_df['CVaR'].values,
                y=results_df['return'].values,
                mode='lines+markers',
                name='효율적 프론티어',
                text=[f"λ={ra:.4f}" for ra in results_df['risk_aversion']],
                hovertemplate="CVaR: %{x:.6f}<br>수익률: %{y:.6f}<br>%{text}<extra></extra>",
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            # Find tangent portfolio (max Sharpe)
            if 'sharpe' in results_df.columns:
                max_idx = results_df['sharpe'].idxmax()
                tangent = results_df.loc[max_idx]
                fig.add_trace(go.Scatter(
                    x=[tangent['CVaR']], y=[tangent['return']],
                    mode='markers',
                    name=f"탄젠트 포트폴리오 (Sharpe={tangent['sharpe']:.4f})",
                    marker=dict(size=14, color='gold', symbol='star', line=dict(width=2, color='black'))
                ))

            # Min variance point
            min_var_idx = results_df['CVaR'].idxmin()
            min_var = results_df.loc[min_var_idx]
            fig.add_trace(go.Scatter(
                x=[min_var['CVaR']], y=[min_var['return']],
                mode='markers',
                name='최소 리스크 포트폴리오',
                marker=dict(size=12, color='green', symbol='diamond')
            ))

        # Plot custom portfolios
        if custom_dict:
            for name, (weights_dict, cash_val) in custom_dict.items():
                try:
                    perf = cvar_utils.evaluate_user_input_portfolios(
                        cvar_optimizer.CVaR(returns_dict=returns_dict, cvar_params=ef_params,
                                            api_settings={"api": "cvxpy"}),
                        {name: (weights_dict, cash_val)}, returns_dict
                    )
                    if perf is not None and len(perf) > 0:
                        row = perf.iloc[0]
                        fig.add_trace(go.Scatter(
                            x=[row['CVaR']], y=[row['return']],
                            mode='markers+text',
                            name=name,
                            text=[name],
                            textposition='top center',
                            marker=dict(size=12, symbol='x', color='red')
                        ))
                except Exception as e:
                    logger.warning("Custom portfolio eval failed for %s: %s", name, e)

        fig.update_layout(
            title="효율적 프론티어 (Mean-CVaR)",
            xaxis_title="CVaR (위험)",
            yaxis_title="기대 수익률",
            height=550,
            template="plotly_white",
            hovermode="closest"
        )

        # Prepare results table
        table_data = []
        if results_df is not None:
            for _, row in results_df.iterrows():
                table_data.append({
                    'risk_aversion': round(float(row['risk_aversion']), 6),
                    'expected_return': round(float(row['return']), 6),
                    'cvar': round(float(row['CVaR']), 6),
                    'objective': round(float(row['obj']), 6),
                    'sharpe': round(float(row.get('sharpe', 0)), 4),
                    'volatility': round(float(row.get('volatility', 0)), 6),
                })

        response = {
            'success': True,
            'frontier_chart': json.loads(plotly.io.to_json(fig)),
            'table_data': table_data,
            'total_elapsed': round(elapsed, 2),
            'num_points': len(table_data),
        }
        return jsonify(response)

    except Exception as e:
        logger.error("Frontier error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
