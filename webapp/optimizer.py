"""
Mean-CVaR Portfolio Optimization Engine (cufolio backend)

Uses NVIDIA's cufolio library from quantitative-portfolio-optimization:
  - cvar_optimizer.CVaR  for LP/MILP optimization (CVXPY + cuOpt/CLARABEL)
  - cvar_utils           for scenario generation (KDE/Gaussian/Historical)
  - backtest             for portfolio backtesting
  - Portfolio, CvarParameters, CvarData  data structures
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from dataclasses import dataclass, field

from cufolio import cvar_optimizer, cvar_utils
from cufolio.cvar_parameters import CvarParameters
from cufolio.cvar_data import CvarData
from cufolio.portfolio import Portfolio
from cufolio import backtest as cufolio_backtest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver detection
# ---------------------------------------------------------------------------
def _detect_solver_settings() -> dict:
    """Detect best available CVXPY solver.  Prefer cuOpt GPU, fallback CLARABEL."""
    solvers = cp.installed_solvers()
    if "CUOPT" in solvers:
        return {
            "solver": cp.CUOPT,
            "verbose": False,
            "solver_method": "PDLP",
            "time_limit": 30,
        }
    return {
        "solver": cp.CLARABEL,
        "verbose": False,
        "tol_gap_abs": 1e-4,
        "tol_gap_rel": 1e-4,
        "tol_feas": 1e-4,
    }


def _detect_kde_device() -> str:
    """Detect whether cuML GPU KDE is available."""
    try:
        import cuml.neighbors  # noqa: F401
        return "GPU"
    except Exception:
        return "CPU"


# ---------------------------------------------------------------------------
# Data classes  (kept for webapp API compatibility)
# ---------------------------------------------------------------------------
@dataclass
class OptimizationParams:
    tickers: list[str]
    start_date: str
    end_date: str
    confidence_level: float = 0.95
    risk_aversion: float = 0.5
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: float | None = None
    risk_free_rate: float = 0.04


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
    cumulative_benchmark: pd.DataFrame
    efficient_frontier: list[dict] = field(default_factory=list)


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
    return_type: str = "log"          # simple / log
    scenario_method: str = "kde"      # historical / kde / gaussian
    num_scenarios: int = 10000
    kde_bandwidth: float = 0.01
    kde_kernel: str = "gaussian"
    # Per-asset bounds
    asset_bounds: list[dict] = field(default_factory=list)
    # Cash allocation
    cash_min: float = 0.0
    cash_max: float = 0.0
    # Leverage
    leverage_target: float = 1.0
    # Turnover
    turnover_target: float | None = None
    current_weights: list[float] | None = None
    # CVaR limit
    cvar_limit: float | None = None
    # Cardinality
    max_assets: int | None = None
    # Backtest
    test_split_date: str | None = None
    benchmark_portfolios: dict[str, list[float]] | None = None


@dataclass
class BacktestResult:
    sharpe: float
    sortino: float
    max_drawdown: float
    annual_return: float
    annual_vol: float
    cumulative: pd.Series
    train_sharpe: float | None = None
    train_sortino: float | None = None
    test_sharpe: float | None = None
    test_sortino: float | None = None
    split_date: str | None = None
    benchmark_metrics: dict | None = None
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_params(params: OptimizationParams) -> None:
    if not (0.90 <= params.confidence_level <= 0.99):
        raise ValueError(
            f"confidence_level must be in [0.90, 0.99], got {params.confidence_level}. "
            "A value of 1.0 causes division by zero in the CVaR formulation."
        )
    if not (0.0 <= params.risk_aversion <= 1.0):
        raise ValueError(f"risk_aversion must be in [0.0, 1.0], got {params.risk_aversion}.")
    if not (-1.0 <= params.min_weight <= 1.0):
        raise ValueError(f"min_weight must be in [-1.0, 1.0], got {params.min_weight}.")
    if not (-1.0 <= params.max_weight <= 1.0):
        raise ValueError(f"max_weight must be in [-1.0, 1.0], got {params.max_weight}.")
    if params.min_weight > params.max_weight:
        raise ValueError(f"min_weight ({params.min_weight}) must be <= max_weight ({params.max_weight}).")
    if len(params.tickers) < 2:
        raise ValueError(f"At least 2 tickers are required, got {len(params.tickers)}.")


# ---------------------------------------------------------------------------
# Data preparation
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


def _build_returns_dict(
    prices: pd.DataFrame,
    return_type: str = "LOG",
    regime_name: str = "webapp",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Convert prices DataFrame to cufolio's returns_dict format."""
    rt = return_type.upper()
    if rt == "LOG":
        returns_df = np.log(prices / prices.shift(1)).dropna()
    else:
        returns_df = (prices / prices.shift(1) - 1).dropna()

    returns_array = returns_df.to_numpy()
    m = np.mean(returns_array, axis=0)
    cov = np.cov(returns_array.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    return {
        "return_type": rt,
        "returns": returns_df,
        "regime": {
            "name": regime_name,
            "range": (
                start_date or str(prices.index[0].date()),
                end_date or str(prices.index[-1].date()),
            ),
        },
        "dates": returns_df.index,
        "mean": m,
        "covariance": cov,
        "tickers": list(returns_df.columns),
    }


def _generate_scenarios(returns_dict: dict, params) -> dict:
    """Generate scenarios using cufolio (supports KDE GPU/CPU, Gaussian, Historical)."""
    method_map = {"historical": "no_fit", "kde": "kde", "gaussian": "gaussian"}
    fit_type = method_map.get(
        getattr(params, "scenario_method", "historical"), "no_fit"
    )

    num_scen = getattr(params, "num_scenarios", 0)
    if num_scen <= 0:
        num_scen = len(returns_dict["returns"])

    kde_device = _detect_kde_device()
    bandwidth = getattr(params, "kde_bandwidth", 0.01)
    kernel = getattr(params, "kde_kernel", "gaussian")

    scenario_settings = {
        "num_scen": num_scen,
        "fit_type": fit_type,
        "kde_settings": {
            "bandwidth": bandwidth,
            "kernel": kernel,
            "device": kde_device,
        },
        "verbose": False,
    }

    try:
        returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_settings)
    except Exception as e:
        # GPU KDE may fail on small GPUs, fallback to CPU
        if kde_device == "GPU":
            logger.warning("GPU KDE failed (%s), falling back to CPU", e)
            scenario_settings["kde_settings"]["device"] = "CPU"
            returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_settings)
        else:
            raise

    return returns_dict


# ---------------------------------------------------------------------------
# cufolio CvarParameters builder
# ---------------------------------------------------------------------------
def _build_cvar_params(params, tickers: list[str]) -> CvarParameters:
    """Build cufolio CvarParameters from webapp params."""
    # Weight bounds
    w_min = getattr(params, "min_weight", 0.0)
    w_max = getattr(params, "max_weight", 1.0)

    # Per-asset bounds
    asset_bounds = getattr(params, "asset_bounds", [])
    if asset_bounds:
        w_min_dict = {}
        w_max_dict = {}
        ab_map = {ab["ticker"]: ab for ab in asset_bounds if "ticker" in ab}
        for t in tickers:
            if t in ab_map:
                w_min_dict[t] = ab_map[t].get("min", w_min)
                w_max_dict[t] = ab_map[t].get("max", w_max)
            else:
                w_min_dict[t] = w_min
                w_max_dict[t] = w_max
        w_min_dict["others"] = w_min
        w_max_dict["others"] = w_max
        w_min = w_min_dict
        w_max = w_max_dict

    # Cash
    c_min = getattr(params, "cash_min", 0.0)
    c_max = getattr(params, "cash_max", 0.0)

    # Leverage
    L_tar = getattr(params, "leverage_target", 1.0)

    # Turnover
    T_tar = getattr(params, "turnover_target", None)

    # CVaR limit
    cvar_limit = getattr(params, "cvar_limit", None)
    if cvar_limit is not None:
        cvar_limit = cvar_limit / 100.0  # webapp sends as %

    # Cardinality
    cardinality = getattr(params, "max_assets", None)

    # Risk aversion and confidence
    risk_aversion = getattr(params, "risk_aversion", 1.0)
    confidence = getattr(params, "confidence_level", 0.95)

    return CvarParameters(
        w_min=w_min,
        w_max=w_max,
        c_min=c_min,
        c_max=c_max,
        L_tar=L_tar,
        T_tar=T_tar,
        cvar_limit=cvar_limit,
        cardinality=cardinality,
        risk_aversion=risk_aversion,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Solve with cufolio
# ---------------------------------------------------------------------------
def _solve_cvar(returns_dict: dict, cvar_params: CvarParameters,
                existing_portfolio: Portfolio | None = None):
    """
    Build and solve CVaR problem using cufolio.
    Returns (result_row, portfolio) from cufolio.
    Tries cuOpt GPU first, falls back to CLARABEL CPU.
    """
    solver_settings = _detect_solver_settings()

    cvar_problem = cvar_optimizer.CVaR(
        returns_dict=returns_dict,
        cvar_params=cvar_params,
        existing_portfolio=existing_portfolio,
    )

    try:
        result_row, portfolio = cvar_problem.solve_optimization_problem(
            solver_settings=solver_settings,
            print_results=False,
        )
    except Exception as e:
        # If cuOpt fails, try CLARABEL
        if solver_settings.get("solver") == cp.CUOPT:
            logger.warning("cuOpt solver failed (%s), falling back to CLARABEL", e)
            fallback = {
                "solver": cp.CLARABEL,
                "verbose": False,
                "tol_gap_abs": 1e-4,
                "tol_gap_rel": 1e-4,
                "tol_feas": 1e-4,
            }
            # Rebuild problem for CLARABEL
            cvar_problem = cvar_optimizer.CVaR(
                returns_dict=returns_dict,
                cvar_params=cvar_params,
                existing_portfolio=existing_portfolio,
            )
            result_row, portfolio = cvar_problem.solve_optimization_problem(
                solver_settings=fallback,
                print_results=False,
            )
        else:
            raise

    return result_row, portfolio, cvar_problem


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------
def _compute_sortino(daily_returns: np.ndarray, rf_daily: float = 0.0) -> float:
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    annual_ret = np.mean(daily_returns) * 252
    return (annual_ret - rf_daily * 252) / downside_std if downside_std > 0 else 0.0


def _compute_max_drawdown(cumulative: pd.Series) -> float:
    running_max = cumulative.cummax()
    dd = (cumulative - running_max) / running_max
    return float(-dd.min() * 100) if len(dd) > 0 else 0.0


# ---------------------------------------------------------------------------
# Basic optimization (backward compatible)
# ---------------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return (prices / prices.shift(1) - 1).dropna()


def generate_scenarios(returns: pd.DataFrame) -> np.ndarray:
    """Each historical day is one scenario."""
    return returns.values


def optimize_cvar(
    scenarios: np.ndarray,
    mean_returns: np.ndarray,
    params: OptimizationParams,
) -> tuple[np.ndarray, float]:
    """
    Wrapper for basic CVaR optimization via cufolio.
    Kept for backward compatibility with frontier.py.
    """
    # Build a minimal returns_dict
    n_assets = len(mean_returns)
    tickers = params.tickers if hasattr(params, "tickers") else [f"A{i}" for i in range(n_assets)]

    # CvarData: R is (n_assets, num_scenarios), scenarios input is (S, N)
    R = scenarios.T  # (N, S)
    S = scenarios.shape[0]
    p = np.ones(S) / S
    cvar_data = CvarData(mean=mean_returns, R=R, p=p)

    cov = np.cov(scenarios.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    returns_dict = {
        "return_type": "LOG",
        "returns": pd.DataFrame(scenarios, columns=tickers),
        "regime": {"name": "basic", "range": (params.start_date, params.end_date)},
        "dates": pd.RangeIndex(S),
        "mean": mean_returns,
        "covariance": cov,
        "tickers": tickers,
        "cvar_data": cvar_data,
    }

    cvar_params = CvarParameters(
        w_min=params.min_weight,
        w_max=params.max_weight,
        c_min=0.0,
        c_max=0.0,
        L_tar=1.0,
        risk_aversion=params.risk_aversion,
        confidence=params.confidence_level,
    )

    _, portfolio, _ = _solve_cvar(returns_dict, cvar_params)

    weights = np.array(portfolio.weights)
    # Compute CVaR from portfolio
    port_scen = scenarios @ weights
    beta = params.confidence_level
    var_threshold = np.percentile(port_scen, (1 - beta) * 100)
    tail = port_scen[port_scen <= var_threshold]
    cvar = float(tail.mean()) if len(tail) > 0 else float(var_threshold)

    return weights, cvar


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


def backtest(
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cumulative returns for optimal portfolio & benchmarks."""
    port_daily = (returns * weights).sum(axis=1)
    cum_port = (1 + port_daily).cumprod()
    eq_daily = returns.mean(axis=1)
    cum_eq = (1 + eq_daily).cumprod()
    cum_indiv = (1 + returns).cumprod()
    benchmark = cum_indiv.copy()
    benchmark["EqualWeight"] = cum_eq
    return cum_port, benchmark


# ---------------------------------------------------------------------------
# Public API: basic optimization
# ---------------------------------------------------------------------------
def run_optimization(params: OptimizationParams) -> OptimizationResult:
    """Full pipeline: data -> scenarios -> optimise -> backtest."""
    validate_params(params)

    prices = fetch_prices(params.tickers, params.start_date, params.end_date)
    if prices.empty:
        raise ValueError("No price data found for the given tickers and date range.")

    returns = compute_returns(prices)
    if len(returns) < 30:
        raise ValueError(f"Insufficient data: only {len(returns)} trading days found. At least 30 required.")

    missing = set(params.tickers) - set(returns.columns)
    if missing:
        raise ValueError(f"No data returned for ticker(s): {', '.join(sorted(missing))}. Check symbols and date range.")

    scenarios = generate_scenarios(returns)
    mean_ret = returns.mean().values

    weights, cvar = optimize_cvar(scenarios, mean_ret, params)
    frontier = compute_efficient_frontier(scenarios, mean_ret, params)
    cum_port, cum_bench = backtest(weights, returns)

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
# ADVANCED CVaR OPTIMIZATION  (Feature 1) — using cufolio
# ===========================================================================

def backtest_advanced(
    weights: np.ndarray,
    cash_weight: float,
    returns: pd.DataFrame,
    params: AdvancedOptimizationParams,
) -> BacktestResult:
    """Comprehensive backtest with Sharpe, Sortino, MaxDD, train/test, benchmarks."""
    rf_daily = params.risk_free_rate / 252

    port_daily = returns.values @ weights
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
        train_mask = np.array(returns.index < split_dt)
        test_mask = np.array(returns.index >= split_dt)

        if train_mask.sum() > 5 and test_mask.sum() > 5:
            train_ret = port_daily[train_mask]
            test_ret = port_daily[test_mask]

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
    """Full advanced pipeline using cufolio: data -> scenarios -> optimize -> backtest."""
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

    # 2. Build returns_dict (cufolio format)
    rt = "LOG" if params.return_type == "log" else "NORMAL"
    returns_dict = _build_returns_dict(
        prices, return_type=rt,
        regime_name="advanced",
        start_date=params.start_date,
        end_date=params.end_date,
    )

    returns_df = returns_dict["returns"]
    if len(returns_df) < 30:
        raise ValueError(f"데이터 부족: {len(returns_df)}일만 확인됨 (최소 30일 필요).")

    historical_returns = returns_df.values

    # 3. Generate scenarios via cufolio (KDE GPU/CPU, Gaussian, Historical)
    returns_dict = _generate_scenarios(returns_dict, params)

    # 4. Build CvarParameters
    cvar_params = _build_cvar_params(params, params.tickers)

    # Build existing portfolio for turnover constraint
    existing_portfolio = None
    if params.turnover_target is not None and params.current_weights is not None:
        if len(params.current_weights) == len(params.tickers):
            existing_portfolio = Portfolio(
                name="current",
                tickers=params.tickers,
                weights=np.array(params.current_weights),
                cash=0.0,
            )

    # 5. Solve via cufolio
    result_row, portfolio, cvar_problem = _solve_cvar(
        returns_dict, cvar_params, existing_portfolio
    )

    weights = np.array(portfolio.weights)
    cash = float(portfolio.cash)

    # Extract metrics from result_row
    expected_return_daily = float(result_row["return"])
    cvar_value = float(result_row["CVaR"])

    # 6. Backtest
    bt = backtest_advanced(weights, cash, returns_df, params)

    # Stats
    mean_ret = returns_dict["mean"]
    port_vol = float(np.std(returns_df.values @ weights, ddof=1) * np.sqrt(252))
    port_ret = float(mean_ret @ weights) * 252
    rf = params.risk_free_rate
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
    rf_daily = rf / 252
    port_daily = returns_df.values @ weights
    sortino = _compute_sortino(port_daily, rf_daily)

    weight_dict = {t: round(float(w), 6) for t, w in zip(params.tickers, weights)}

    # Get scenarios for chart display
    cvar_data = returns_dict.get("cvar_data")
    if cvar_data is not None:
        scenarios = cvar_data.R.T  # (S, N) for charts
    else:
        scenarios = historical_returns

    return AdvancedOptimizationResult(
        weights=weight_dict,
        cash_weight=round(cash, 6),
        expected_return=round(port_ret * 100, 2),
        cvar=round(cvar_value * 100, 4),
        volatility=round(port_vol * 100, 2),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=bt.max_drawdown,
        confidence_level=params.confidence_level,
        prices_df=prices,
        returns_df=returns_df,
        scenarios=scenarios,
        historical_returns=historical_returns,
        backtest=bt,
        params=params,
    )
