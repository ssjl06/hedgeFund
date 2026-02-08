"""
Efficient Frontier Analysis with Logarithmic Lambda Sweep

Mirrors NVIDIA's efficient_frontier.ipynb:
  - Logarithmic risk-aversion sweep
  - Custom portfolio overlay evaluation
  - CSV export of frontier data
"""

import io
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from optimizer import (
    OptimizationParams, fetch_prices, optimize_cvar,
)
from scenarios import ReturnType, ScenarioMethod, compute_returns, generate_scenarios


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
    min_exp: float = -3.0       # 10^(-3) = 0.001
    max_exp: float = 1.0        # 10^1 = 10
    n_steps: int = 50
    # Scenario settings
    return_type: str = "simple"
    scenario_method: str = "historical"
    num_scenarios: int = 0
    kde_bandwidth: float = 0.5
    kde_kernel: str = "gaussian"
    # Custom portfolio for overlay
    custom_portfolio: dict[str, float] = field(default_factory=dict)  # ticker -> weight


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
    tangent_idx: int              # index of best Sharpe point
    custom_eval: CustomPortfolioEval | None
    tickers: list[str]
    prices_df: pd.DataFrame
    returns_df: pd.DataFrame
    scenarios: np.ndarray


def compute_logarithmic_frontier(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: FrontierParams,
) -> list[FrontierPoint]:
    """
    Sweep lambda on a logarithmic scale, then normalise to [0, 1]
    for the CVaR LP formulation.

    Raw lambdas from logspace: 10^min_exp to 10^max_exp
    Normalised: lam_norm = raw / (1 + raw), mapping (0, inf) -> (0, 1)
    """
    raw_lambdas = np.logspace(params.min_exp, params.max_exp, params.n_steps)
    points = []

    for raw_lam in raw_lambdas:
        lam_norm = raw_lam / (1.0 + raw_lam)
        # Clamp to valid range
        lam_norm = max(0.01, min(0.99, lam_norm))

        p = OptimizationParams(
            tickers=params.tickers,
            start_date=params.start_date,
            end_date=params.end_date,
            confidence_level=params.confidence_level,
            risk_aversion=lam_norm,
            min_weight=params.min_weight,
            max_weight=params.max_weight,
        )
        try:
            w, cvar = optimize_cvar(scenarios, mean_returns, p)
            port_ret = float(mean_returns @ w) * 252
            port_vol = float(np.std(scenarios @ w, ddof=1) * np.sqrt(252))
            points.append(FrontierPoint(
                lambda_val=round(float(raw_lam), 6),
                expected_return=round(port_ret * 100, 2),
                cvar=round(cvar * 100, 4),
                volatility=round(port_vol * 100, 2),
                weights={t: round(float(wi), 6)
                         for t, wi in zip(params.tickers, w)},
            ))
        except RuntimeError:
            continue

    return points


def evaluate_custom_portfolio(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: FrontierParams,
) -> CustomPortfolioEval | None:
    """Evaluate CVaR/return/vol for user-provided weights."""
    if not params.custom_portfolio:
        return None

    N = len(params.tickers)
    weights = np.zeros(N)
    for i, t in enumerate(params.tickers):
        weights[i] = params.custom_portfolio.get(t, 0.0)

    # Normalise if they don't sum to 1
    wsum = weights.sum()
    if abs(wsum) < 1e-10:
        return None
    if abs(wsum - 1.0) > 1e-6:
        weights = weights / wsum

    port_ret = float(mean_returns @ weights) * 252
    port_vol = float(np.std(scenarios @ weights, ddof=1) * np.sqrt(252))

    # CVaR
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
        weights={t: round(float(w), 6) for t, w in zip(params.tickers, weights)},
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
    """Full frontier pipeline."""
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

    rt = ReturnType(params.return_type)
    returns = compute_returns(prices, rt)
    if len(returns) < 30:
        raise ValueError(f"데이터 부족: {len(returns)}일 (최소 30일 필요).")

    sm = ScenarioMethod(params.scenario_method)
    scenarios = generate_scenarios(
        returns, sm, params.num_scenarios,
        params.kde_bandwidth, params.kde_kernel,
    )
    mean_ret = returns.mean().values

    # Frontier
    points = compute_logarithmic_frontier(scenarios, mean_ret, params)
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
    custom_eval = evaluate_custom_portfolio(scenarios, mean_ret, params)

    return FrontierResult(
        points=points,
        tangent_idx=best_idx,
        custom_eval=custom_eval,
        tickers=params.tickers,
        prices_df=prices,
        returns_df=returns,
        scenarios=scenarios,
    )
