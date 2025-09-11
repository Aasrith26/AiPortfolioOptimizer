# data_preparer.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("data_preparer")

def _winsorize(series: pd.Series, zmax: float = 8.0) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return series.fillna(0.0)
    lo, hi = mu - zmax * sd, mu + zmax * sd
    return series.clip(lo, hi)

def sanitize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure finite, well‑behaved daily returns suitable for cov/optimization."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(how="any", inplace=True)
    out = out.apply(_winsorize)
    var = out.var(ddof=0)
    keep = var[var > 0].index
    dropped = [c for c in out.columns if c not in keep]
    if dropped:
        logger.warning("sanitize_returns(): dropped zero‑variance assets: %s", dropped)
    out = out[keep].astype(np.float64)
    return out

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        logger.error("calculate_returns(): empty prices")
        return pd.DataFrame()
    raw = prices.pct_change().iloc[1:]
    clean = sanitize_returns(raw)
    logger.info("calculate_returns(): cleaned returns shape=%s", clean.shape)
    return clean
