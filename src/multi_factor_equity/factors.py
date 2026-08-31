"""Cross-sectional 12-1 momentum and realized-vol factors.

All windows are trailing (right-aligned). Nothing here looks forward in
time; a future price spike cannot change a score at an earlier date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MOM_LOOKBACK = 252
MOM_SKIP = 21
VOL_WINDOW = 63


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. First row is NaN."""
    if prices.empty:
        return prices.copy()
    return prices.pct_change()


def momentum_12_1(
    prices: pd.DataFrame,
    lookback: int = MOM_LOOKBACK,
    skip: int = MOM_SKIP,
) -> pd.DataFrame:
    """12-1 momentum: skip the most recent skip days, use lookback.

    At date t: P[t - skip] / P[t - lookback] - 1.
    Classic Jegadeesh-Titman 12-1 on daily data (~252/21).
    """
    if lookback <= skip:
        raise ValueError("lookback must be > skip")
    lagged = prices.shift(skip)
    start = prices.shift(lookback)
    return lagged / start - 1.0


def compounded_return(
    returns: pd.DataFrame,
    lookback: int,
    skip: int = 0,
) -> pd.DataFrame:
    """Trailing compounded return from t-lookback to t-skip (exclusive of skip)."""
    if lookback <= skip:
        raise ValueError("lookback must be > skip")
    log1p = np.log1p(returns)
    window = lookback - skip
    rolled = log1p.rolling(window=window, min_periods=window).sum()
    if skip:
        rolled = rolled.shift(skip)
    return np.expm1(rolled)


def realized_vol(
    returns: pd.DataFrame,
    window: int = VOL_WINDOW,
) -> pd.DataFrame:
    """Trailing realized volatility (daily std, not annualized)."""
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across names. sklearn StandardScaler (ddof=0).

    A name that is NaN on a date is left NaN and excluded from that
    date's mean/std. Dates with <2 finite names stay all-NaN.
    """
    values = panel.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    scaler = StandardScaler()
    for i in range(values.shape[0]):
        row = values[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        subset = row[mask].reshape(-1, 1)
        if float(subset.std(ddof=0)) < 1e-12:
            continue
        out[i, mask] = scaler.fit_transform(subset).ravel()
    return pd.DataFrame(out, index=panel.index, columns=panel.columns)


def residualize_equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    """Subtract the equal-weight cross-sectional mean (beta=1 market).

    Simple neutralization helper: r_i - mean_j r_j. Does not estimate
    trailing betas.
    """
    mkt = returns.mean(axis=1)
    return returns.sub(mkt, axis=0)


def combined_score(
    prices: pd.DataFrame,
    lookback: int = MOM_LOOKBACK,
    skip: int = MOM_SKIP,
    vol_window: int = VOL_WINDOW,
    mom_weight: float = 0.5,
    vol_weight: float = 0.5,
    neutralize_returns: bool = False,
) -> pd.DataFrame:
    """0.5 * z(12-1 mom) + 0.5 * z(-vol) unless weights are overridden.

    Low realized vol is the high score (inverse-vol). If
    neutralize_returns is True, both legs are built from equal-weight
    residual returns rather than raw prices / raw returns.
    """
    rets = daily_returns(prices)
    if neutralize_returns:
        rets = residualize_equal_weight(rets)
        mom = compounded_return(rets, lookback=lookback, skip=skip)
        vol = realized_vol(rets, window=vol_window)
    else:
        mom = momentum_12_1(prices, lookback=lookback, skip=skip)
        vol = realized_vol(rets, window=vol_window)
    mom_z = cross_sectional_zscore(mom)
    vol_z = cross_sectional_zscore(vol)
    return mom_weight * mom_z + vol_weight * (-vol_z)


def score_asof(
    prices_upto: pd.DataFrame,
    **kwargs,
) -> pd.Series:
    """Combined score at the last date of a trailing price window.

    Callers must pass prices.loc[:t]; this function never reads past
    the last index label it is given.
    """
    if prices_upto.empty:
        return pd.Series(dtype=float)
    frame = combined_score(prices_upto, **kwargs)
    return frame.iloc[-1]
