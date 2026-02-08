"""
Scenario Generation for CVaR Portfolio Optimization (cufolio backend)

Uses cufolio's cvar_utils.generate_cvar_data() which supports:
  - KDE with cuML GPU or sklearn CPU
  - Gaussian multivariate normal
  - Historical (no_fit)

This module provides backward-compatible enums and wrappers.
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
    bandwidth: float = 0.01,
    kernel: str = "gaussian",
    seed: int = 42,
) -> np.ndarray:
    """
    Generate return scenarios using cufolio backend.

    Parameters
    ----------
    returns : DataFrame of historical returns (T x N)
    method : ScenarioMethod enum
    num_scenarios : number of synthetic scenarios (0 = use len(returns))
    bandwidth : KDE bandwidth parameter
    kernel : KDE kernel type
    seed : random seed for reproducibility

    Returns
    -------
    scenarios : ndarray of shape (S, N)
    """
    from cufolio import cvar_utils

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
        # Use cufolio's KDE (GPU if available, else CPU)
        try:
            import cuml.neighbors  # noqa: F401
            kde_device = "GPU"
        except Exception:
            kde_device = "CPU"

        kde_settings = {
            "bandwidth": bandwidth,
            "kernel": kernel,
            "device": kde_device,
        }

        try:
            samples = cvar_utils.generate_samples_kde(
                n_scen, data, kde_settings=kde_settings, verbose=False
            )
        except Exception:
            # Fallback to CPU
            kde_settings["device"] = "CPU"
            samples = cvar_utils.generate_samples_kde(
                n_scen, data, kde_settings=kde_settings, verbose=False
            )
        return samples

    if method == ScenarioMethod.GAUSSIAN:
        rng = np.random.default_rng(seed)
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        cov = (cov + cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            cov -= np.eye(n_assets) * eigvals.min() * 1.1
        return rng.multivariate_normal(mean, cov, size=n_scen)

    raise ValueError(f"Unknown scenario method: {method}")
