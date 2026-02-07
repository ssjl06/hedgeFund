"""
Mean-CVaR Portfolio Optimization Engine

Implements the Rockafellar & Uryasev (2000) linear programming formulation
of Conditional Value-at-Risk portfolio optimization.

Workflow mirrors NVIDIA's cvar_basic.ipynb:
  1. Data preparation  - download prices, compute returns
  2. Scenario generation - historical returns as CVaR scenarios
  3. Optimization        - solve Mean-CVaR LP
  4. Backtest            - evaluate portfolio performance
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import linprog
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OptimizationParams:
    tickers: list[str]
    start_date: str
    end_date: str
    confidence_level: float = 0.95       # beta for CVaR
    risk_aversion: float = 0.5           # lambda  (0=min-risk, 1=max-return)
    min_weight: float = 0.0              # per-asset lower bound
    max_weight: float = 1.0              # per-asset upper bound
    target_return: float | None = None   # optional target return constraint
    risk_free_rate: float = 0.04         # annualised risk-free rate


@dataclass
class OptimizationResult:
    weights: dict[str, float]
    expected_return: float
    cvar: float
    volatility: float
    sharpe_ratio: float
    confidence_level: float
    prices_df: pd.DataFrame
    returns_df: pd.DataFrame
    cumulative_portfolio: pd.Series
    cumulative_benchmark: pd.DataFrame   # equal-weight & individual
    efficient_frontier: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_params(params: OptimizationParams) -> None:
    """Validate optimization parameters. Raises ValueError on bad input."""
    if not (0.90 <= params.confidence_level <= 0.99):
        raise ValueError(
            f"confidence_level must be in [0.90, 0.99], got {params.confidence_level}. "
            "A value of 1.0 causes division by zero in the CVaR formulation."
        )
    if not (0.0 <= params.risk_aversion <= 1.0):
        raise ValueError(
            f"risk_aversion must be in [0.0, 1.0], got {params.risk_aversion}."
        )
    if not (-1.0 <= params.min_weight <= 1.0):
        raise ValueError(
            f"min_weight must be in [-1.0, 1.0], got {params.min_weight}."
        )
    if not (-1.0 <= params.max_weight <= 1.0):
        raise ValueError(
            f"max_weight must be in [-1.0, 1.0], got {params.max_weight}."
        )
    if params.min_weight > params.max_weight:
        raise ValueError(
            f"min_weight ({params.min_weight}) must be <= max_weight ({params.max_weight})."
        )
    if len(params.tickers) < 2:
        raise ValueError(
            f"At least 2 tickers are required, got {len(params.tickers)}."
        )


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, timeout=30)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    df = df.dropna()
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return (prices / prices.shift(1) - 1).dropna()


# ---------------------------------------------------------------------------
# 2. Scenario generation  (historical simulation)
# ---------------------------------------------------------------------------
def generate_scenarios(returns: pd.DataFrame) -> np.ndarray:
    """Each historical day is one scenario.  Shape: (S, N)."""
    return returns.values


# ---------------------------------------------------------------------------
# 3. Mean-CVaR optimisation  (Rockafellar-Uryasev LP)
# ---------------------------------------------------------------------------
def optimize_cvar(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: OptimizationParams,
) -> tuple[np.ndarray, float]:
    """
    Solve:
        max   lambda * mu'w  -  (1-lambda) * CVaR_beta(w)

    Reformulated as LP  (minimisation):
        min  -(lambda) * mu'w  +  (1-lambda) * [ alpha + 1/((1-beta)*S) * sum z_s ]

        s.t.  z_s  >=  -(r_s' w) - alpha       for all s
              z_s  >=  0                         for all s
              sum w_i = 1
              lo <= w_i <= hi

    Decision variables:  x = [w_1..w_N, alpha, z_1..z_S]

    Returns (weights, cvar) where cvar is derived from the LP solution.
    """
    S, N = scenarios.shape
    beta = params.confidence_level
    lam = params.risk_aversion

    # number of variables
    n_vars = N + 1 + S  # weights + alpha + auxiliary z

    # ---- objective ----
    c = np.zeros(n_vars)
    # weights part:  -lambda * mu
    c[:N] = -lam * mean_returns
    # alpha part:  (1-lambda) * 1
    c[N] = (1 - lam)
    # z part:  (1-lambda) / ((1-beta)*S)
    c[N + 1:] = (1 - lam) / ((1 - beta) * S)

    # ---- inequality constraints:  z_s >= -r_s'w - alpha  ->  r_s'w + alpha + z_s >= 0
    #      linprog format:  A_ub @ x <= b_ub   ->  -(r_s'w + alpha + z_s) <= 0
    A_ub = np.zeros((S, n_vars))
    A_ub[:, :N] = -scenarios          # -r_s * w
    A_ub[:, N] = -1.0                 # -alpha
    for s in range(S):
        A_ub[s, N + 1 + s] = -1.0    # -z_s
    b_ub = np.zeros(S)

    # ---- equality: sum w_i = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # optional target-return constraint:  mu'w >= target  ->  -mu'w <= -target
    if params.target_return is not None:
        row = np.zeros((1, n_vars))
        row[0, :N] = -mean_returns
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, -params.target_return)

    # ---- bounds ----
    bounds = []
    for _ in range(N):
        bounds.append((params.min_weight, params.max_weight))
    bounds.append((None, None))  # alpha unbounded
    for _ in range(S):
        bounds.append((0, None))  # z_s >= 0

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x[:N]

    # compute CVaR directly from the LP solution variables
    alpha = result.x[N]
    z_values = result.x[N + 1:]
    cvar = alpha + z_values.sum() / ((1 - beta) * S)

    return weights, cvar


# ---------------------------------------------------------------------------
# 4. Efficient frontier
# ---------------------------------------------------------------------------
def compute_efficient_frontier(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: OptimizationParams,
    n_points: int = 30,
) -> list[dict]:
    """Trace frontier by varying risk_aversion lambda from 0 to 1."""
    frontier = []
    for lam in np.linspace(0.01, 0.99, n_points):
        p = OptimizationParams(
            tickers=params.tickers,
            start_date=params.start_date,
            end_date=params.end_date,
            confidence_level=params.confidence_level,
            risk_aversion=lam,
            min_weight=params.min_weight,
            max_weight=params.max_weight,
        )
        try:
            w, cvar = optimize_cvar(scenarios, mean_returns, p)
            port_ret = float(mean_returns @ w) * 252
            port_vol = float(np.sqrt((scenarios @ w).var()) * np.sqrt(252))
            frontier.append({
                "lambda": round(lam, 3),
                "return": round(port_ret * 100, 2),
                "cvar": round(cvar * 100, 4),
                "volatility": round(port_vol * 100, 2),
            })
        except RuntimeError:
            continue
    return frontier


# ---------------------------------------------------------------------------
# 5. Backtest
# ---------------------------------------------------------------------------
def backtest(
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cumulative returns for optimal portfolio & benchmarks."""
    port_daily = (returns * weights).sum(axis=1)
    cum_port = (1 + port_daily).cumprod()

    # equal-weight benchmark
    eq_daily = returns.mean(axis=1)
    cum_eq = (1 + eq_daily).cumprod()

    # individual assets
    cum_indiv = (1 + returns).cumprod()

    benchmark = cum_indiv.copy()
    benchmark["EqualWeight"] = cum_eq

    return cum_port, benchmark


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_optimization(params: OptimizationParams) -> OptimizationResult:
    """Full pipeline: data -> scenarios -> optimise -> backtest."""

    # 0. validate
    validate_params(params)

    # 1. data
    prices = fetch_prices(params.tickers, params.start_date, params.end_date)

    if prices.empty:
        raise ValueError("No price data found for the given tickers and date range.")

    returns = compute_returns(prices)

    if len(returns) < 30:
        raise ValueError(
            f"Insufficient data: only {len(returns)} trading days found. "
            "At least 30 required."
        )

    # check for tickers dropped entirely
    missing = set(params.tickers) - set(returns.columns)
    if missing:
        raise ValueError(
            f"No data returned for ticker(s): {', '.join(sorted(missing))}. "
            "Check symbols and date range."
        )

    # 2. scenarios
    scenarios = generate_scenarios(returns)
    mean_ret = returns.mean().values

    # 3. optimise
    weights, cvar = optimize_cvar(scenarios, mean_ret, params)

    # 4. efficient frontier
    frontier = compute_efficient_frontier(scenarios, mean_ret, params)

    # 5. backtest
    cum_port, cum_bench = backtest(weights, returns)

    # summary stats
    port_return = float(mean_ret @ weights) * 252
    port_vol = float(np.sqrt((scenarios @ weights).var()) * np.sqrt(252))
    rf = params.risk_free_rate
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0

    weight_dict = {t: round(float(w), 6) for t, w in zip(params.tickers, weights)}

    return OptimizationResult(
        weights=weight_dict,
        expected_return=round(port_return * 100, 2),
        cvar=round(cvar * 100, 4),
        volatility=round(port_vol * 100, 2),
        sharpe_ratio=round(sharpe, 4),
        confidence_level=params.confidence_level,
        prices_df=prices,
        returns_df=returns,
        cumulative_portfolio=cum_port,
        cumulative_benchmark=cum_bench,
        efficient_frontier=frontier,
    )
