"""
Efficient Frontier Analysis with Logarithmic Lambda Sweep (cufolio backend)

Uses cufolio's cvar_optimizer.CVaR for each lambda point,
matching NVIDIA's efficient_frontier.ipynb approach.
"""

import io
import logging
import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass, field

from cufolio import cvar_optimizer, cvar_utils
from cufolio.cvar_parameters import CvarParameters
from cufolio.cvar_data import CvarData

from optimizer import (
    fetch_prices, _build_returns_dict, _generate_scenarios,
    _detect_solver_settings,
)

logger = logging.getLogger(__name__)


@dataclass
class FrontierParams:
    tickers: list[str]
    start_date: str
    end_date: str
    confidence_level: float = 0.95
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_rate: float = 0.04
    # Frontier sweep settings
    min_exp: float = -3.0
    max_exp: float = 1.0
    n_steps: int = 50
    # Scenario settings
    return_type: str = "log"
    scenario_method: str = "kde"
    num_scenarios: int = 10000
    kde_bandwidth: float = 0.01
    kde_kernel: str = "gaussian"
    # Custom portfolio for overlay
    custom_portfolio: dict[str, float] = field(default_factory=dict)


@dataclass
class FrontierPoint:
    lambda_val: float
    expected_return: float    # annualised %
    cvar: float               # daily %
    volatility: float         # annualised %
    weights: dict[str, float]


@dataclass
class CustomPortfolioEval:
    expected_return: float
    cvar: float
    volatility: float
    sharpe: float
    weights: dict[str, float]


@dataclass
class FrontierResult:
    points: list[FrontierPoint]
    tangent_idx: int
    custom_eval: CustomPortfolioEval | None
    tickers: list[str]
    prices_df: pd.DataFrame
    returns_df: pd.DataFrame
    scenarios: np.ndarray


def compute_logarithmic_frontier(
    returns_dict: dict,
    params: FrontierParams,
) -> list[FrontierPoint]:
    """
    Sweep lambda on a logarithmic scale using cufolio's CVaR optimizer,
    matching NVIDIA's efficient_frontier.ipynb approach.

    Raw lambdas from logspace: 10^min_exp to 10^max_exp (reversed: high to low)
    """
    risk_aversion_list = np.logspace(
        params.min_exp, params.max_exp, params.n_steps
    )[::-1]  # high to low like NVIDIA's code

    solver_settings = _detect_solver_settings()

    # Build base CvarParameters
    cvar_params = CvarParameters(
        w_min=params.min_weight,
        w_max=params.max_weight,
        c_min=0.0,
        c_max=0.0,
        L_tar=1.0,
        risk_aversion=risk_aversion_list[0],
        confidence=params.confidence_level,
    )

    # Build the CVaR problem once, then sweep risk_aversion
    try:
        cvar_problem = cvar_optimizer.CVaR(
            returns_dict=returns_dict,
            cvar_params=cvar_params,
        )
    except Exception as e:
        logger.warning("Failed to create CVaR problem: %s", e)
        raise RuntimeError(f"CVaR 문제 생성 실패: {e}")

    points = []
    tickers = returns_dict["tickers"]
    covariance = returns_dict["covariance"]

    for i, ra_value in enumerate(risk_aversion_list):
        try:
            cvar_problem.params.update_risk_aversion(ra_value)
            cvar_problem.risk_aversion_param.value = ra_value

            result_row, portfolio = cvar_problem.solve_optimization_problem(
                solver_settings=solver_settings,
                print_results=False,
            )

            expected_return = float(result_row["return"])
            cvar_value = float(result_row["CVaR"])
            variance = portfolio.calculate_portfolio_variance(covariance)
            volatility = np.sqrt(variance)

            weights_dict = {
                t: round(float(w), 6)
                for t, w in zip(tickers, portfolio.weights)
            }

            points.append(FrontierPoint(
                lambda_val=round(float(ra_value), 6),
                expected_return=round(expected_return * 252 * 100, 2),
                cvar=round(cvar_value * 100, 4),
                volatility=round(float(volatility) * np.sqrt(252) * 100, 2),
                weights=weights_dict,
            ))

        except Exception as e:
            logger.debug("Frontier point failed at ra=%.4f: %s", ra_value, e)
            # Try fallback solver for this point
            if solver_settings.get("solver") == cp.CUOPT:
                try:
                    fallback = {
                        "solver": cp.CLARABEL, "verbose": False,
                        "tol_gap_abs": 1e-4, "tol_gap_rel": 1e-4, "tol_feas": 1e-4,
                    }
                    result_row, portfolio = cvar_problem.solve_optimization_problem(
                        solver_settings=fallback, print_results=False,
                    )
                    expected_return = float(result_row["return"])
                    cvar_value = float(result_row["CVaR"])
                    variance = portfolio.calculate_portfolio_variance(covariance)
                    volatility = np.sqrt(variance)
                    weights_dict = {
                        t: round(float(w), 6)
                        for t, w in zip(tickers, portfolio.weights)
                    }
                    points.append(FrontierPoint(
                        lambda_val=round(float(ra_value), 6),
                        expected_return=round(expected_return * 252 * 100, 2),
                        cvar=round(cvar_value * 100, 4),
                        volatility=round(float(volatility) * np.sqrt(252) * 100, 2),
                        weights=weights_dict,
                    ))
                except Exception:
                    continue
            else:
                continue

    return points


def evaluate_custom_portfolio(
    returns_dict: dict,
    params: FrontierParams,
) -> CustomPortfolioEval | None:
    """Evaluate CVaR/return/vol for user-provided weights."""
    if not params.custom_portfolio:
        return None

    tickers = returns_dict["tickers"]
    N = len(tickers)
    weights = np.zeros(N)
    for i, t in enumerate(tickers):
        weights[i] = params.custom_portfolio.get(t, 0.0)

    wsum = weights.sum()
    if abs(wsum) < 1e-10:
        return None
    if abs(wsum - 1.0) > 1e-6:
        weights = weights / wsum

    # Use cufolio's evaluate_portfolio_performance if cvar_data is available
    cvar_data = returns_dict.get("cvar_data")
    if cvar_data is not None:
        scenarios = cvar_data.R.T  # (S, N)
        mean_returns = cvar_data.mean
    else:
        returns_df = returns_dict["returns"]
        scenarios = returns_df.values
        mean_returns = returns_dict["mean"]

    port_ret = float(mean_returns @ weights) * 252
    port_vol = float(np.std(scenarios @ weights, ddof=1) * np.sqrt(252))

    beta = params.confidence_level
    port_scen = scenarios @ weights
    var_threshold = np.percentile(port_scen, (1 - beta) * 100)
    tail = port_scen[port_scen <= var_threshold]
    cvar = float(tail.mean()) if len(tail) > 0 else float(var_threshold)

    rf = params.risk_free_rate
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0

    return CustomPortfolioEval(
        expected_return=round(port_ret * 100, 2),
        cvar=round(cvar * 100, 4),
        volatility=round(port_vol * 100, 2),
        sharpe=round(sharpe, 4),
        weights={t: round(float(w), 6) for t, w in zip(tickers, weights)},
    )


def export_frontier_csv(points: list[FrontierPoint], tickers: list[str]) -> str:
    """Export frontier data as CSV string."""
    buf = io.StringIO()
    cols = ["lambda", "expected_return_pct", "cvar_pct", "volatility_pct"] + tickers
    buf.write(",".join(cols) + "\n")
    for pt in points:
        row = [str(pt.lambda_val), str(pt.expected_return), str(pt.cvar), str(pt.volatility)]
        for t in tickers:
            row.append(str(pt.weights.get(t, 0.0)))
        buf.write(",".join(row) + "\n")
    return buf.getvalue()


def run_frontier_analysis(params: FrontierParams) -> FrontierResult:
    """Full frontier pipeline using cufolio."""
    if len(params.tickers) < 2:
        raise ValueError("종목을 2개 이상 입력해주세요.")
    if not (0.90 <= params.confidence_level <= 0.99):
        raise ValueError("신뢰수준은 0.90~0.99 범위여야 합니다.")

    # Data
    prices = fetch_prices(params.tickers, params.start_date, params.end_date)
    if prices.empty:
        raise ValueError("주가 데이터를 찾을 수 없습니다.")

    missing = set(params.tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"데이터를 찾을 수 없는 종목: {', '.join(sorted(missing))}")

    rt = "LOG" if params.return_type == "log" else "NORMAL"
    returns_dict = _build_returns_dict(
        prices, return_type=rt,
        regime_name="frontier",
        start_date=params.start_date,
        end_date=params.end_date,
    )

    returns_df = returns_dict["returns"]
    if len(returns_df) < 30:
        raise ValueError(f"데이터 부족: {len(returns_df)}일 (최소 30일 필요).")

    # Generate scenarios
    returns_dict = _generate_scenarios(returns_dict, params)

    # Frontier
    points = compute_logarithmic_frontier(returns_dict, params)
    if not points:
        raise RuntimeError("프론티어 계산에 실패했습니다.")

    # Tangent portfolio (best Sharpe)
    rf = params.risk_free_rate
    best_idx = 0
    best_sharpe = -np.inf
    for i, pt in enumerate(points):
        s = (pt.expected_return / 100.0 - rf) / (pt.volatility / 100.0) if pt.volatility > 0 else 0
        if s > best_sharpe:
            best_sharpe = s
            best_idx = i

    # Custom portfolio evaluation
    custom_eval = evaluate_custom_portfolio(returns_dict, params)

    # Get scenarios for charts
    cvar_data = returns_dict.get("cvar_data")
    if cvar_data is not None:
        scenarios = cvar_data.R.T
    else:
        scenarios = returns_df.values

    return FrontierResult(
        points=points,
        tangent_idx=best_idx,
        custom_eval=custom_eval,
        tickers=params.tickers,
        prices_df=prices,
        returns_df=returns_df,
        scenarios=scenarios,
    )
