# optimizer.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("optimizer")

def _nearest_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    vals[vals < eps] = eps
    return (vecs * vals) @ vecs.T

def _safe_cov(returns: pd.DataFrame, min_periods: int = 30, ridge: float = 1e-8) -> pd.DataFrame:
    """Finite, PSD covariance with small ridge."""
    X = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    cov = X.cov(min_periods=min_periods).values
    if not np.all(np.isfinite(cov)):
        logger.warning("safe_cov(): non‑finite entries in sample cov; cleaning")
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    n = cov.shape[0]
    cov += np.eye(n) * ridge
    cov = _nearest_psd(cov, eps=ridge)
    return pd.DataFrame(cov, index=X.columns, columns=X.columns)

# ---- integrate into your existing entry points ----

def get_optimal_portfolio(returns: pd.DataFrame, expected_returns: pd.Series, objective: str = "Sharpe"):
    """Use safe covariance in the core solver."""
    mu = expected_returns.reindex(returns.columns).fillna(0.0).astype(np.float64)
    cov = _safe_cov(returns)
    # Call your pre‑existing internal routine that expects mu (Series) and cov (DataFrame)
    return _solve_core(mu, cov, objective)

def get_portfolio_by_slider(returns: pd.DataFrame, expected_returns: pd.Series, slider: float):
    """Same safe covariance path for slider‑based frontier point."""
    mu = expected_returns.reindex(returns.columns).fillna(0.0).astype(np.float64)
    cov = _safe_cov(returns)
    return _solve_core_slider(mu, cov, slider)

# Keep your original core implementations under these names or adapt wrappers
def _solve_core(mu: pd.Series, cov: pd.DataFrame, objective: str):
    # import and delegate to your original solver code (unchanged)
    return solve_core(mu, cov, objective)

def _solve_core_slider(mu: pd.Series, cov: pd.DataFrame, slider: float):
    return solve_core_slider(mu, cov, slider)
