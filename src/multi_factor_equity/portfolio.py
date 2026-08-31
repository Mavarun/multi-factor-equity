"""Long-short weights, turnover, and costed portfolio returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROSS_EXPOSURE = 1.0  # |long| + |short| = 1 (0.5 each side when dollar-neutral)


def long_short_weights(
    scores: pd.Series,
    method: str = "quantile",
    quantile: float = 0.2,
) -> pd.Series:
    """Map a cross-section of scores to dollar-neutral long-short weights.

    method="quantile": equal-weight the top/bottom quantile (at least one
    name each side). Gross exposure is 1.0 (0.5 long, 0.5 short).
    method="rank": rank-weighted, demeaned, scaled to gross 1.0.
    NaN scores are excluded (weight 0).
    """
    s = scores.astype(float)
    valid = s.dropna()
    w = pd.Series(0.0, index=s.index, dtype=float)
    if valid.size < 4:
        return w
    if method == "quantile":
        q = min(max(quantile, 1.0 / valid.size), 0.5)
        n_side = max(1, int(np.floor(q * valid.size)))
        ranked = valid.sort_values(ascending=False)
        long_names = ranked.index[:n_side]
        short_names = ranked.index[-n_side:]
        # Guard against overlap when n is tiny.
        overlap = set(long_names) & set(short_names)
        if overlap:
            long_names = ranked.index[: max(1, n_side - len(overlap))]
            short_names = ranked.index[-max(1, n_side - len(overlap)) :]
            overlap = set(long_names) & set(short_names)
            if overlap:
                return w
        w.loc[long_names] = 0.5 / len(long_names)
        w.loc[short_names] = -0.5 / len(short_names)
    elif method == "rank":
        ranks = valid.rank(method="average")
        demeaned = ranks - ranks.mean()
        denom = demeaned.abs().sum()
        if denom < 1e-12:
            return w
        w.loc[valid.index] = demeaned / denom
    else:
        raise ValueError(f"unknown method: {method}")
    return w


def two_way_turnover(prev: pd.Series, new: pd.Series) -> float:
    """sum(|w_new - w_prev|). Full book flip of a 1.0-gross LS book is 2.0."""
    aligned = pd.concat(
        [prev.astype(float).fillna(0.0), new.astype(float).fillna(0.0)],
        axis=1,
    ).fillna(0.0)
    aligned.columns = ["prev", "new"]
    return float((aligned["new"] - aligned["prev"]).abs().sum())


def costed_portfolio_returns(
    asset_returns: pd.DataFrame,
    weight_panel: pd.DataFrame,
    cost_drag: pd.Series | None = None,
    cost_bps: float = 0.0,
    turnover_by_date: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Daily gross and net portfolio returns.

    weight_panel.loc[t] is the book held through date t (set at the
    previous close). Gross = sum_i w_{i,t} * r_{i,t}.
    If cost_drag is given it is subtracted as-is. Otherwise costs are
    cost_bps/1e4 * turnover_by_date (0 on dates with no trade).
    """
    cols = asset_returns.columns.union(weight_panel.columns)
    rets = asset_returns.reindex(columns=cols).fillna(0.0)
    w = weight_panel.reindex(index=rets.index, columns=cols).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    if cost_drag is None:
        if turnover_by_date is None:
            cost_drag = pd.Series(0.0, index=gross.index)
        else:
            cost_drag = (cost_bps / 1e4) * turnover_by_date.reindex(
                gross.index
            ).fillna(0.0)
    net = gross - cost_drag.reindex(gross.index).fillna(0.0)
    return gross, net


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe. NaN if fewer than 2 finite observations."""
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.size < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd < 1e-16:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def cumulative_return(returns: pd.Series) -> float:
    r = returns.astype(float).fillna(0.0)
    if r.empty:
        return 0.0
    return float(np.expm1(np.log1p(r).sum()))
