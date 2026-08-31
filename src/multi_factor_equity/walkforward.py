"""Month-end walk-forward: trailing scores only, OOS book, Spearman IC."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multi_factor_equity.factors import (
    MOM_LOOKBACK,
    MOM_SKIP,
    VOL_WINDOW,
    daily_returns,
    score_asof,
)
from multi_factor_equity.portfolio import (
    costed_portfolio_returns,
    cumulative_return,
    long_short_weights,
    sharpe,
    two_way_turnover,
)


def month_end_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Last available session of each calendar month in index."""
    s = pd.Series(1, index=pd.DatetimeIndex(index))
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).apply(lambda x: x.index.max()))


def _spearman(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a.astype(float), b.astype(float)], axis=1).dropna()
    if aligned.shape[0] < 3:
        return float("nan")
    x = aligned.iloc[:, 0].to_numpy()
    y = aligned.iloc[:, 1].to_numpy()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def _pooled_ic(pairs: list[tuple[pd.Series, pd.Series]]) -> float:
    if not pairs:
        return float("nan")
    xs, ys = [], []
    for score, fwd in pairs:
        aligned = pd.concat([score.astype(float), fwd.astype(float)], axis=1).dropna()
        xs.append(aligned.iloc[:, 0].to_numpy())
        ys.append(aligned.iloc[:, 1].to_numpy())
    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


@dataclass
class RebalanceRecord:
    date: pd.Timestamp
    next_date: pd.Timestamp
    scores: pd.Series
    weights: pd.Series
    forward_returns: pd.Series
    ic: float
    turnover: float


@dataclass
class WalkForwardResult:
    rebalances: list[RebalanceRecord]
    daily_gross: pd.Series
    daily_net: pd.Series
    daily_weights: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.metrics)


def walk_forward(
    prices: pd.DataFrame,
    cost_bps: float = 10.0,
    method: str = "quantile",
    quantile: float = 0.2,
    lookback: int = MOM_LOOKBACK,
    skip: int = MOM_SKIP,
    vol_window: int = VOL_WINDOW,
    neutralize_returns: bool = False,
    is_frac: float = 0.5,
    min_names: int = 4,
) -> WalkForwardResult:
    """Month-end rebalance using only prices up to each decision date.

    At rebalance t the score is ``score_asof(prices.loc[:t])``. The book
    decided at close t earns daily returns from t+1 through the next
    month-end (inclusive). Spearman IC uses the holding-period return
    P[next]/P[t] - 1 vs the score at t.

    Pooled IS IC uses the first ``is_frac`` of rebalances; OOS IC uses
    the rest. Construction is trailing on both halves — the split is
    about evaluation windows, not a fitted parameter.
    """
    prices = prices.sort_index()
    prices = prices.loc[:, prices.notna().any()]
    rets = daily_returns(prices)
    month_ends = month_end_dates(prices.index)
    # Need lookback history before the first decision.
    loc = prices.index.get_indexer(month_ends, method="pad")
    eligible = []
    for d, i in zip(month_ends, loc):
        if i >= lookback:
            eligible.append(pd.Timestamp(d))
    if len(eligible) < 3:
        raise ValueError(
            f"need >=3 month-end rebalances after lookback={lookback}, got {len(eligible)}"
        )

    score_kwargs = dict(
        lookback=lookback,
        skip=skip,
        vol_window=vol_window,
        neutralize_returns=neutralize_returns,
    )

    records: list[RebalanceRecord] = []
    prev_w = pd.Series(0.0, index=prices.columns)
    weight_panel = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover_by_date = pd.Series(0.0, index=prices.index)
    prev_reb: pd.Timestamp | None = None

    for k, t in enumerate(eligible[:-1]):
        next_t = eligible[k + 1]
        hist = prices.loc[:t]
        scores = score_asof(hist, **score_kwargs)
        n_finite = int(scores.notna().sum())
        if n_finite < min_names:
            continue
        weights = long_short_weights(scores, method=method, quantile=quantile)
        weights = weights.reindex(prices.columns).fillna(0.0)

        # Book held through dates (prev_reb, t] is prev_w (0 before first).
        if prev_reb is None:
            held = prices.index[(prices.index <= t)]
            # first trade at close t; hold zeros into t
            weight_panel.loc[held] = 0.0
        else:
            held = prices.index[(prices.index > prev_reb) & (prices.index <= t)]
            if len(held):
                weight_panel.loc[held, :] = prev_w.values

        to = two_way_turnover(prev_w, weights)
        turnover_by_date.loc[t] = to

        px_t = prices.loc[t]
        px_n = prices.loc[next_t]
        fwd = px_n / px_t - 1.0
        ic = _spearman(scores, fwd)

        records.append(
            RebalanceRecord(
                date=t,
                next_date=next_t,
                scores=scores,
                weights=weights,
                forward_returns=fwd,
                ic=ic,
                turnover=to,
            )
        )
        prev_w = weights
        prev_reb = t

    if prev_reb is not None:
        tail = prices.index[prices.index > prev_reb]
        if len(tail):
            weight_panel.loc[tail, :] = prev_w.values

    if not records:
        raise ValueError("no valid rebalances")

    gross, net = costed_portfolio_returns(
        rets,
        weight_panel,
        cost_bps=cost_bps,
        turnover_by_date=turnover_by_date,
    )

    n = len(records)
    split = max(1, min(n - 1, int(np.floor(is_frac * n))))
    is_recs = records[:split]
    oos_recs = records[split:]
    oos_start = oos_recs[0].date if oos_recs else records[-1].date

    def _mean_abs_ic(recs: list[RebalanceRecord]) -> float:
        vals = [r.ic for r in recs if np.isfinite(r.ic)]
        return float(np.mean(np.abs(vals))) if vals else float("nan")

    oos_gross = gross.loc[gross.index > oos_start]
    oos_net = net.loc[net.index > oos_start]
    # Daily series after first OOS decision; include from the day after
    # the last IS rebalance so the first OOS month is counted.
    metrics = {
        "n_rebalances": n,
        "n_is": len(is_recs),
        "n_oos": len(oos_recs),
        "is_end": str(is_recs[-1].date.date()) if is_recs else None,
        "oos_start": str(oos_start.date()),
        "gross_sharpe": sharpe(oos_gross),
        "net_sharpe": sharpe(oos_net),
        "gross_sharpe_full": sharpe(gross),
        "net_sharpe_full": sharpe(net),
        "mean_turnover": float(np.mean([r.turnover for r in oos_recs]))
        if oos_recs
        else float("nan"),
        "mean_turnover_full": float(np.mean([r.turnover for r in records])),
        "mean_abs_ic_oos": _mean_abs_ic(oos_recs),
        "mean_abs_ic_is": _mean_abs_ic(is_recs),
        "ic_is_pooled": _pooled_ic([(r.scores, r.forward_returns) for r in is_recs]),
        "ic_oos_pooled": _pooled_ic([(r.scores, r.forward_returns) for r in oos_recs]),
        "ic_pooled_all": _pooled_ic([(r.scores, r.forward_returns) for r in records]),
        "cum_gross_oos": cumulative_return(oos_gross),
        "cum_net_oos": cumulative_return(oos_net),
        "cum_gross_full": cumulative_return(gross),
        "cum_net_full": cumulative_return(net),
        "cost_bps": cost_bps,
        "method": method,
        "quantile": quantile,
        "lookback": lookback,
        "skip": skip,
        "vol_window": vol_window,
        "neutralize_returns": neutralize_returns,
    }
    return WalkForwardResult(
        rebalances=records,
        daily_gross=gross,
        daily_net=net,
        daily_weights=weight_panel,
        metrics=metrics,
    )
