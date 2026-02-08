"""
Scenario Generation for CVaR Portfolio Optimization

Supports three methods mirroring NVIDIA's cvar_basic.ipynb:
  - HISTORICAL: raw return matrix (baseline)
  - KDE: Kernel Density Estimation via sklearn
  - GAUSSIAN: parametric multivariate normal sampling
"""

from enum import Enum

import numpy as np
import pandas as pd


class ReturnType(Enum):
    SIMPLE = "simple"
    LOG = "log"


class ScenarioMethod(Enum):
    HISTORICAL = "historical"
    KDE = "kde"
    GAUSSIAN = "gaussian"


def compute_returns(prices: pd.DataFrame, return_type: ReturnType = ReturnType.SIMPLE) -> pd.DataFrame:
    """Compute daily returns from price data."""
    if return_type == ReturnType.LOG:
        return np.log(prices / prices.shift(1)).dropna()
    return (prices / prices.shift(1) - 1).dropna()


def generate_scenarios(
    returns: pd.DataFrame,
    method: ScenarioMethod = ScenarioMethod.HISTORICAL,
    num_scenarios: int = 0,
    bandwidth: float = 0.5,
    kernel: str = "gaussian",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate return scenarios for CVaR optimization.

    Parameters
    ----------
    returns : DataFrame of historical returns (T x N)
    method : ScenarioMethod enum
    num_scenarios : number of synthetic scenarios (0 = use len(returns))
    bandwidth : KDE bandwidth parameter
    kernel : KDE kernel type (gaussian, tophat, epanechnikov, etc.)
    seed : random seed for reproducibility

    Returns
    -------
    scenarios : ndarray of shape (S, N)
    """
    data = returns.values
    n_obs, n_assets = data.shape
    n_scen = num_scenarios if num_scenarios > 0 else n_obs

    if method == ScenarioMethod.HISTORICAL:
        if n_scen == n_obs:
            return data
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_obs, size=n_scen, replace=True)
        return data[idx]

    if method == ScenarioMethod.KDE:
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(data)
        samples = kde.sample(n_scen, random_state=seed)
        return samples

    if method == ScenarioMethod.GAUSSIAN:
        rng = np.random.default_rng(seed)
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        # Ensure positive semi-definiteness
        cov = (cov + cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            cov -= np.eye(n_assets) * eigvals.min() * 1.1
        return rng.multivariate_normal(mean, cov, size=n_scen)

    raise ValueError(f"Unknown scenario method: {method}")
