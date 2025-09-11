# forecaster.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("forecaster")

def generate_forecasted_returns(
    returns: pd.DataFrame,
    sentiments: dict,
    lookback: int = 60,
    sentiment_scale: float = 0.10
) -> pd.Series:
    """Blend recent mean/vol with bounded sentiment to get finite expected returns."""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    recent = returns.tail(lookback)
    mu = recent.mean().astype(np.float64)
    sigma = recent.std(ddof=0).astype(np.float64)
    sigma = sigma.replace(0, np.nan)

    # Internal z‑signal (bounded) and external sentiment (bounded)
    z = (mu / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2, 2) * sentiment_scale
    s = pd.Series(sentiments or {}, dtype=float).reindex(mu.index).fillna(0.0).clip(-1, 1)

    # Combine, keep tilt modest, and build strictly finite μ_adj
    tilt = z.add(s, fill_value=0.0).clip(-0.2, 0.2)
    mu_adj = (mu + tilt * sigma.fillna(0.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    logger.info("generate_forecasted_returns(): forecasts ready for %d assets", mu_adj.size)
    return mu_adj
