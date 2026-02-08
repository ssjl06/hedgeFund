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
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from dataclasses import dataclass, field

from scenarios import ReturnType, ScenarioMethod, compute_returns as sc_compute_returns, generate_scenarios as sc_generate_scenarios


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


# ===========================================================================
# ADVANCED CVaR OPTIMIZATION  (Feature 1)
# ===========================================================================

@dataclass
class AdvancedOptimizationParams:
    tickers: list[str]
    start_date: str
    end_date: str
    confidence_level: float = 0.95
    risk_aversion: float = 0.5
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_rate: float = 0.04
    # Scenario settings
    return_type: str = "simple"       # simple / log
    scenario_method: str = "historical"  # historical / kde / gaussian
    num_scenarios: int = 0            # 0 = use all historical observations
    kde_bandwidth: float = 0.5
    kde_kernel: str = "gaussian"
    # Per-asset bounds: [{"ticker": "AAPL", "min": 0.0, "max": 0.3}, ...]
    asset_bounds: list[dict] = field(default_factory=list)
    # Cash allocation
    cash_min: float = 0.0
    cash_max: float = 0.0
    # Leverage
    leverage_target: float = 1.0      # sum of |weights| target
    # Turnover
    turnover_target: float | None = None
    current_weights: list[float] | None = None
    # CVaR limit
    cvar_limit: float | None = None   # maximum allowable CVaR (daily %)
    # Cardinality
    max_assets: int | None = None     # triggers MILP
    # Backtest
    test_split_date: str | None = None
    benchmark_portfolios: dict[str, list[float]] | None = None  # name -> weights


@dataclass
class BacktestResult:
    sharpe: float
    sortino: float
    max_drawdown: float
    annual_return: float
    annual_vol: float
    cumulative: pd.Series
    # Train/test split results
    train_sharpe: float | None = None
    train_sortino: float | None = None
    test_sharpe: float | None = None
    test_sortino: float | None = None
    split_date: str | None = None
    # Benchmark comparison
    benchmark_metrics: dict | None = None   # name -> {sharpe, sortino, max_dd, cumulative}
    cumulative_benchmarks: pd.DataFrame | None = None


@dataclass
class AdvancedOptimizationResult:
    weights: dict[str, float]
    cash_weight: float
    expected_return: float
    cvar: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    confidence_level: float
    prices_df: pd.DataFrame
    returns_df: pd.DataFrame
    scenarios: np.ndarray
    historical_returns: np.ndarray
    backtest: BacktestResult
    params: AdvancedOptimizationParams


def _compute_sortino(daily_returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """Sortino ratio: excess return / downside deviation."""
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    annual_ret = np.mean(daily_returns) * 252
    return (annual_ret - rf_daily * 252) / downside_std if downside_std > 0 else 0.0


def _compute_max_drawdown(cumulative: pd.Series) -> float:
    """Maximum drawdown as a positive percentage."""
    running_max = cumulative.cummax()
    dd = (cumulative - running_max) / running_max
    return float(-dd.min() * 100) if len(dd) > 0 else 0.0


def optimize_cvar_advanced(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: AdvancedOptimizationParams,
) -> tuple[np.ndarray, float, float]:
    """
    Extended Mean-CVaR LP with cash, leverage, turnover, and CVaR limit.

    Decision variables: x = [w_1..w_N, cash, alpha, z_1..z_S, (t_1..t_N if turnover)]

    Returns (weights_array, cash_weight, cvar).
    """
    S, N = scenarios.shape
    beta = params.confidence_level
    lam = params.risk_aversion
    has_cash = params.cash_max > 0
    has_turnover = (params.turnover_target is not None and
                    params.current_weights is not None and
                    len(params.current_weights) == N)

    # Variable layout: [w(N), cash(1), alpha(1), z(S), t(N if turnover)]
    n_cash = 1
    n_turnover = N if has_turnover else 0
    n_vars = N + n_cash + 1 + S + n_turnover

    idx_cash = N
    idx_alpha = N + 1
    idx_z = N + 2
    idx_t = idx_z + S  # only used if has_turnover

    # ---- Objective: min -lam * mu'w + (1-lam) * [alpha + 1/((1-beta)*S) * sum(z)] ----
    c = np.zeros(n_vars)
    c[:N] = -lam * mean_returns
    c[idx_alpha] = (1 - lam)
    c[idx_z:idx_z + S] = (1 - lam) / ((1 - beta) * S)

    # ---- Inequality constraints ----
    ub_rows = []
    ub_rhs = []

    # z_s >= -r_s'w - alpha  =>  -r_s'w - alpha - z_s <= 0
    A_cvar = np.zeros((S, n_vars))
    A_cvar[:, :N] = -scenarios
    A_cvar[:, idx_alpha] = -1.0
    for s in range(S):
        A_cvar[s, idx_z + s] = -1.0
    ub_rows.append(A_cvar)
    ub_rhs.append(np.zeros(S))

    # CVaR limit: alpha + 1/((1-beta)*S) * sum(z) <= cvar_limit
    if params.cvar_limit is not None:
        row = np.zeros((1, n_vars))
        row[0, idx_alpha] = 1.0
        row[0, idx_z:idx_z + S] = 1.0 / ((1 - beta) * S)
        ub_rows.append(row)
        ub_rhs.append(np.array([params.cvar_limit / 100.0]))  # convert from % to decimal

    # Turnover: |w_i - w_i_current| <= t_i, sum(t_i) <= turnover_target
    if has_turnover:
        cur_w = np.array(params.current_weights)
        # w_i - cur_w_i <= t_i  =>  w_i - t_i <= cur_w_i
        A_t1 = np.zeros((N, n_vars))
        for i in range(N):
            A_t1[i, i] = 1.0
            A_t1[i, idx_t + i] = -1.0
        ub_rows.append(A_t1)
        ub_rhs.append(cur_w)
        # -(w_i - cur_w_i) <= t_i  =>  -w_i - t_i <= -cur_w_i
        A_t2 = np.zeros((N, n_vars))
        for i in range(N):
            A_t2[i, i] = -1.0
            A_t2[i, idx_t + i] = -1.0
        ub_rows.append(A_t2)
        ub_rhs.append(-cur_w)
        # sum(t_i) <= turnover_target
        row = np.zeros((1, n_vars))
        row[0, idx_t:idx_t + N] = 1.0
        ub_rows.append(row)
        ub_rhs.append(np.array([params.turnover_target]))

    A_ub = np.vstack(ub_rows) if ub_rows else None
    b_ub = np.concatenate(ub_rhs) if ub_rhs else None

    # ---- Equality: sum(w) + cash = leverage_target ----
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    A_eq[0, idx_cash] = 1.0
    b_eq = np.array([params.leverage_target])

    # ---- Bounds ----
    # Per-asset bounds
    asset_bound_map = {}
    for ab in params.asset_bounds:
        asset_bound_map[ab.get("ticker", "")] = (ab.get("min", params.min_weight),
                                                   ab.get("max", params.max_weight))
    bounds = []
    for i, ticker in enumerate(params.tickers):
        lo, hi = asset_bound_map.get(ticker, (params.min_weight, params.max_weight))
        bounds.append((lo, hi))
    # Cash bounds
    bounds.append((params.cash_min, params.cash_max) if has_cash else (0.0, 0.0))
    # Alpha unbounded
    bounds.append((None, None))
    # z >= 0
    for _ in range(S):
        bounds.append((0.0, None))
    # t >= 0 (turnover aux)
    for _ in range(n_turnover):
        bounds.append((0.0, None))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"Advanced optimization failed: {result.message}")

    weights = result.x[:N]
    cash = result.x[idx_cash]
    alpha = result.x[idx_alpha]
    z_vals = result.x[idx_z:idx_z + S]
    cvar = alpha + z_vals.sum() / ((1 - beta) * S)

    return weights, float(cash), float(cvar)


def optimize_cvar_milp(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: AdvancedOptimizationParams,
) -> tuple[np.ndarray, float, float]:
    """
    MILP cardinality-constrained CVaR optimization using scipy.optimize.milp.

    Additional binary variables b_i: w_i <= max * b_i, sum(b_i) <= max_assets.
    """
    S, N = scenarios.shape
    beta = params.confidence_level
    lam = params.risk_aversion
    max_assets = params.max_assets
    has_cash = params.cash_max > 0

    # Variables: [w(N), cash(1), alpha(1), z(S), b(N)]
    n_vars = N + 1 + 1 + S + N
    idx_cash = N
    idx_alpha = N + 1
    idx_z = N + 2
    idx_b = idx_z + S

    # ---- Objective ----
    c = np.zeros(n_vars)
    c[:N] = -lam * mean_returns
    c[idx_alpha] = (1 - lam)
    c[idx_z:idx_z + S] = (1 - lam) / ((1 - beta) * S)

    # ---- Constraints via LinearConstraint ----
    constraint_rows = []

    # CVaR: z_s >= -r_s'w - alpha  =>  r_s'w + alpha + z_s >= 0
    A_cvar = np.zeros((S, n_vars))
    A_cvar[:, :N] = scenarios
    A_cvar[:, idx_alpha] = 1.0
    for s in range(S):
        A_cvar[s, idx_z + s] = 1.0
    constraint_rows.append(LinearConstraint(A_cvar, lb=0.0, ub=np.inf))

    # Budget: sum(w) + cash = leverage_target
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    A_eq[0, idx_cash] = 1.0
    constraint_rows.append(LinearConstraint(A_eq, lb=params.leverage_target, ub=params.leverage_target))

    # Cardinality linking: w_i <= max_weight * b_i  =>  w_i - max_weight * b_i <= 0
    A_card = np.zeros((N, n_vars))
    for i in range(N):
        A_card[i, i] = 1.0
        A_card[i, idx_b + i] = -params.max_weight
    constraint_rows.append(LinearConstraint(A_card, lb=-np.inf, ub=0.0))

    # Also need: w_i >= min_weight * b_i  =>  w_i - min_weight * b_i >= 0
    if params.min_weight > 0:
        A_card_lo = np.zeros((N, n_vars))
        for i in range(N):
            A_card_lo[i, i] = 1.0
            A_card_lo[i, idx_b + i] = -params.min_weight
        constraint_rows.append(LinearConstraint(A_card_lo, lb=0.0, ub=np.inf))

    # sum(b_i) <= max_assets
    A_max = np.zeros((1, n_vars))
    A_max[0, idx_b:idx_b + N] = 1.0
    constraint_rows.append(LinearConstraint(A_max, lb=0, ub=max_assets))

    # CVaR limit
    if params.cvar_limit is not None:
        A_cl = np.zeros((1, n_vars))
        A_cl[0, idx_alpha] = 1.0
        A_cl[0, idx_z:idx_z + S] = 1.0 / ((1 - beta) * S)
        constraint_rows.append(LinearConstraint(A_cl, lb=-np.inf, ub=params.cvar_limit / 100.0))

    # ---- Bounds ----
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    # Weights
    for i in range(N):
        lb[i] = params.min_weight
        ub[i] = params.max_weight
    # Cash
    lb[idx_cash] = params.cash_min if has_cash else 0.0
    ub[idx_cash] = params.cash_max if has_cash else 0.0
    # Alpha unbounded
    lb[idx_alpha] = -np.inf
    # z >= 0 (already 0)
    # Binary b: 0 or 1
    for i in range(N):
        lb[idx_b + i] = 0.0
        ub[idx_b + i] = 1.0

    bounds_obj = Bounds(lb=lb, ub=ub)

    # Integer constraints: binary for b variables
    integrality = np.zeros(n_vars)
    integrality[idx_b:idx_b + N] = 1  # 1 = integer (binary due to 0/1 bounds)

    result = milp(c, constraints=constraint_rows, integrality=integrality, bounds=bounds_obj)

    if not result.success:
        raise RuntimeError(f"MILP optimization failed: {result.message}")

    weights = result.x[:N]
    cash = result.x[idx_cash]
    alpha = result.x[idx_alpha]
    z_vals = result.x[idx_z:idx_z + S]
    cvar = alpha + z_vals.sum() / ((1 - beta) * S)

    return weights, float(cash), float(cvar)


def backtest_advanced(
    weights: np.ndarray,
    cash_weight: float,
    returns: pd.DataFrame,
    params: AdvancedOptimizationParams,
) -> BacktestResult:
    """Comprehensive backtest with Sharpe, Sortino, MaxDD, train/test, benchmarks."""
    rf_daily = params.risk_free_rate / 252

    port_daily = (returns.values @ weights)
    cum = pd.Series((1 + port_daily).cumprod(), index=returns.index)

    annual_ret = float(np.mean(port_daily) * 252)
    annual_vol = float(np.std(port_daily, ddof=1) * np.sqrt(252))
    sharpe = (annual_ret - params.risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
    sortino = _compute_sortino(port_daily, rf_daily)
    max_dd = _compute_max_drawdown(cum)

    # Train/test split
    train_sharpe = test_sharpe = train_sortino = test_sortino = None
    split_date = params.test_split_date
    if split_date:
        split_dt = pd.Timestamp(split_date)
        train_mask = returns.index < split_dt
        test_mask = returns.index >= split_dt

        if train_mask.sum() > 5 and test_mask.sum() > 5:
            train_ret = port_daily[train_mask.values]
            test_ret = port_daily[test_mask.values]

            train_vol = float(np.std(train_ret, ddof=1) * np.sqrt(252))
            train_ann = float(np.mean(train_ret) * 252)
            train_sharpe = (train_ann - params.risk_free_rate) / train_vol if train_vol > 0 else 0.0
            train_sortino = _compute_sortino(train_ret, rf_daily)

            test_vol = float(np.std(test_ret, ddof=1) * np.sqrt(252))
            test_ann = float(np.mean(test_ret) * 252)
            test_sharpe = (test_ann - params.risk_free_rate) / test_vol if test_vol > 0 else 0.0
            test_sortino = _compute_sortino(test_ret, rf_daily)

    # Benchmark comparison
    benchmark_metrics = {}
    cum_benchmarks = pd.DataFrame(index=returns.index)
    cum_benchmarks["Optimal"] = cum

    # Equal weight benchmark
    eq_daily = returns.values.mean(axis=1)
    cum_eq = pd.Series((1 + eq_daily).cumprod(), index=returns.index)
    eq_vol = float(np.std(eq_daily, ddof=1) * np.sqrt(252))
    eq_ann = float(np.mean(eq_daily) * 252)
    benchmark_metrics["동일비중"] = {
        "sharpe": round((eq_ann - params.risk_free_rate) / eq_vol if eq_vol > 0 else 0.0, 4),
        "sortino": round(_compute_sortino(eq_daily, rf_daily), 4),
        "max_drawdown": round(_compute_max_drawdown(cum_eq), 2),
        "annual_return": round(eq_ann * 100, 2),
    }
    cum_benchmarks["동일비중"] = cum_eq

    # Custom benchmarks
    if params.benchmark_portfolios:
        for name, bw in params.benchmark_portfolios.items():
            bw_arr = np.array(bw)
            if len(bw_arr) == len(params.tickers):
                b_daily = returns.values @ bw_arr
                b_cum = pd.Series((1 + b_daily).cumprod(), index=returns.index)
                b_vol = float(np.std(b_daily, ddof=1) * np.sqrt(252))
                b_ann = float(np.mean(b_daily) * 252)
                benchmark_metrics[name] = {
                    "sharpe": round((b_ann - params.risk_free_rate) / b_vol if b_vol > 0 else 0.0, 4),
                    "sortino": round(_compute_sortino(b_daily, rf_daily), 4),
                    "max_drawdown": round(_compute_max_drawdown(b_cum), 2),
                    "annual_return": round(b_ann * 100, 2),
                }
                cum_benchmarks[name] = b_cum

    return BacktestResult(
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_drawdown=round(max_dd, 2),
        annual_return=round(annual_ret * 100, 2),
        annual_vol=round(annual_vol * 100, 2),
        cumulative=cum,
        train_sharpe=round(train_sharpe, 4) if train_sharpe is not None else None,
        train_sortino=round(train_sortino, 4) if train_sortino is not None else None,
        test_sharpe=round(test_sharpe, 4) if test_sharpe is not None else None,
        test_sortino=round(test_sortino, 4) if test_sortino is not None else None,
        split_date=split_date,
        benchmark_metrics=benchmark_metrics,
        cumulative_benchmarks=cum_benchmarks,
    )


def run_advanced_optimization(params: AdvancedOptimizationParams) -> AdvancedOptimizationResult:
    """Full advanced pipeline: data -> scenarios -> optimize -> backtest."""
    # Validate basics
    if len(params.tickers) < 2:
        raise ValueError("종목을 2개 이상 입력해주세요.")
    if not (0.90 <= params.confidence_level <= 0.99):
        raise ValueError("신뢰수준은 0.90~0.99 범위여야 합니다.")

    # 1. Data
    prices = fetch_prices(params.tickers, params.start_date, params.end_date)
    if prices.empty:
        raise ValueError("주가 데이터를 찾을 수 없습니다.")

    missing = set(params.tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"데이터를 찾을 수 없는 종목: {', '.join(sorted(missing))}")

    # 2. Returns
    rt = ReturnType(params.return_type)
    returns = sc_compute_returns(prices, rt)
    if len(returns) < 30:
        raise ValueError(f"데이터 부족: {len(returns)}일만 확인됨 (최소 30일 필요).")

    # 3. Scenarios
    sm = ScenarioMethod(params.scenario_method)
    scenarios = sc_generate_scenarios(
        returns, sm, params.num_scenarios,
        params.kde_bandwidth, params.kde_kernel,
    )
    mean_ret = returns.mean().values
    historical_returns = returns.values

    # 4. Optimize
    if params.max_assets is not None and params.max_assets < len(params.tickers):
        weights, cash, cvar = optimize_cvar_milp(scenarios, mean_ret, params)
    else:
        weights, cash, cvar = optimize_cvar_advanced(scenarios, mean_ret, params)

    # 5. Backtest
    bt = backtest_advanced(weights, cash, returns, params)

    # Stats
    port_vol = float(np.std(returns.values @ weights, ddof=1) * np.sqrt(252))
    port_ret = float(mean_ret @ weights) * 252
    rf = params.risk_free_rate
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    rf_daily = rf / 252
    port_daily = returns.values @ weights
    sortino = _compute_sortino(port_daily, rf_daily)

    weight_dict = {t: round(float(w), 6) for t, w in zip(params.tickers, weights)}

    return AdvancedOptimizationResult(
        weights=weight_dict,
        cash_weight=round(cash, 6),
        expected_return=round(port_ret * 100, 2),
        cvar=round(cvar * 100, 4),
        volatility=round(port_vol * 100, 2),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=bt.max_drawdown,
        confidence_level=params.confidence_level,
        prices_df=prices,
        returns_df=returns,
        scenarios=scenarios,
        historical_returns=historical_returns,
        backtest=bt,
        params=params,
    )
