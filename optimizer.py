# optimizer.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("optimizer")

# Profile caps mirrored from app policy
PROFILE_MAX_CAP = {"MinRisk": 0.30, "Sharpe": 0.35, "MaxRet": 0.40}

# ---------- helpers: projections, covariance, PSD ----------

def _project_simplex_caps(w: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Project weights onto {w: sum w=1, 0<=w<=caps} with simple room-proportional fill."""
    w = np.clip(w, 0.0, caps)
    s = w.sum()
    if s <= 0.0:
        # spread across caps if all zero
        total_cap = caps.sum()
        return (caps / total_cap) if total_cap > 0 else np.full_like(w, 1.0 / len(w))
    if abs(s - 1.0) < 1e-12:
        return w
    if s > 1.0:
        w = w / s
        w = np.minimum(w, caps)
        s2 = w.sum()
        return (w / s2) if s2 > 0 else w
    # s < 1.0: distribute remaining mass by available room
    rem = 1.0 - s
    room = caps - w
    mask = room > 1e-12
    if mask.any():
        w[mask] += room[mask] / room[mask].sum() * rem
    else:
        w = w / w.sum()
    return w

def _nearest_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    vals[vals < eps] = eps
    return (vecs * vals) @ vecs.T

def _safe_cov(returns: pd.DataFrame, min_periods: int = 30, ridge: float = 1e-8) -> pd.DataFrame:
    """Finite, PSD covariance with a tiny ridge for numerical stability."""
    X = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    cov = X.cov(min_periods=min_periods).values
    if not np.all(np.isfinite(cov)):
        logger.warning("safe_cov(): non-finite entries in sample cov; cleaning")
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    n = cov.shape
    cov += np.eye(n) * ridge
    cov = _nearest_psd(cov, eps=ridge)
    return pd.DataFrame(cov, index=X.columns, columns=X.columns)

def _perf(mu: pd.Series, cov: pd.DataFrame, w: np.ndarray) -> tuple[float, float, float]:
    ret = float(mu.values @ w) * 252.0
    vol = float(np.sqrt(max(0.0, w @ cov.values @ w)) * np.sqrt(252.0))
    sr = ret / vol if vol > 0 else 0.0
    return ret, vol, sr

# ---------- simple, dependency-free solvers ----------

def _solve_minvar(cov: pd.DataFrame, caps: np.ndarray, iters: int = 500) -> np.ndarray:
    """Projected gradient descent on min w^T Σ w with 0<=w<=caps, sum w=1."""
    n = cov.shape
    w = np.full(n, 1.0 / n, dtype=np.float64)
    lr = 0.5
    G = cov.values
    for _ in range(iters):
        grad = 2.0 * (G @ w)
        w = w - lr * grad
        w = _project_simplex_caps(w, caps)
        lr *= 0.995
    return w

def _solve_maxret(mu: pd.Series, caps: np.ndarray) -> np.ndarray:
    """Greedy maximize μ·w under simplex+caps."""
    vals = mu.values
    order = np.argsort(-vals)
    w = np.zeros_like(vals, dtype=np.float64)
    remaining = 1.0
    for i in order:
        add = min(caps[i], remaining)
        w[i] = add
        remaining -= add
        if remaining <= 1e-12:
            break
    if remaining > 1e-12:
        room = caps - w
        mask = room > 1e-12
        if mask.any():
            w[mask] += room[mask] / room[mask].sum() * remaining
    return _project_simplex_caps(w, caps)

def _solve_best_sharpe(mu: pd.Series, cov: pd.DataFrame, caps: np.ndarray) -> np.ndarray:
    """Grid blend between min-var and max-ret to maximize Sharpe."""
    w_min = _solve_minvar(cov, caps)
    w_max = _solve_maxret(mu, caps)
    best_w, best_sr = None, -1e18
    for a in np.linspace(0.0, 1.0, 51):
        w = _project_simplex_caps((1 - a) * w_min + a * w_max, caps)
        _, _, sr = _perf(mu, cov, w)
        if sr > best_sr:
            best_sr, best_w = sr, w
    return best_w

# ---------- public API expected by app.py ----------

def get_optimal_portfolio(returns: pd.DataFrame, expected_returns: pd.Series, objective: str = "Sharpe"):
    """Compute weights and simple performance using safe covariance; no external core needed."""
    mu = expected_returns.reindex(returns.columns).fillna(0.0).astype(np.float64)
    cov = _safe_cov(returns)
    n = mu.size
    caps = np.full(n, PROFILE_MAX_CAP.get(objective, 0.35), dtype=np.float64)

    if objective == "MinRisk":
        w = _solve_minvar(cov, caps)
    elif objective == "MaxRet":
        w = _solve_maxret(mu, caps)
    else:  # "Sharpe" default
        w = _solve_best_sharpe(mu, cov, caps)

    ret, vol, sr = _perf(mu, cov, w)
    weights = pd.Series(w, index=mu.index, name="Weight")
    perf = {"return": ret, "vol": vol, "sharpe": sr}
    return weights, perf

def get_portfolio_by_slider(returns: pd.DataFrame, expected_returns: pd.Series, slider: float):
    """Blend along the min-var to max-ret segment using slider ∈ [0,100]."""
    mu = expected_returns.reindex(returns.columns).fillna(0.0).astype(np.float64)
    cov = _safe_cov(returns)
    n = mu.size
    caps = np.full(n, 0.35, dtype=np.float64)

    w_min = _solve_minvar(cov, caps)
    w_max = _solve_maxret(mu, caps)
    a = float(np.clip(slider, 0.0, 100.0)) / 100.0
    w = _project_simplex_caps((1 - a) * w_min + a * w_max, caps)

    ret, vol, sr = _perf(mu, cov, w)
    weights = pd.Series(w, index=mu.index, name="Weight")
    perf = {"return": ret, "vol": vol, "sharpe": sr}
    return weights, perf
