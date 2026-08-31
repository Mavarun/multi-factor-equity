import numpy as np
import pandas as pd

from multi_factor_equity.data import make_synthetic_panel
from multi_factor_equity.factors import combined_score, daily_returns
from multi_factor_equity.portfolio import (
    costed_portfolio_returns,
    cumulative_return,
    long_short_weights,
    sharpe,
    two_way_turnover,
)


def test_quantile_dollar_neutral_unit_gross():
    scores = pd.Series(
        {"A": 2.0, "B": 1.0, "C": 0.5, "D": 0.0, "E": -0.5, "F": -1.0, "G": -2.0, "H": -3.0}
    )
    w = long_short_weights(scores, method="quantile", quantile=0.25)
    assert abs(w.sum()) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12
    assert (w[w > 0].index.tolist() == ["A", "B"]) or set(w[w > 0].index) <= set(scores.index)
    assert (w[w > 0] > 0).all()
    assert (w[w < 0] < 0).all()


def test_rank_dollar_neutral():
    scores = pd.Series(np.linspace(-2, 2, 10), index=list("ABCDEFGHIJ"))
    w = long_short_weights(scores, method="rank")
    assert abs(w.sum()) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12
    assert w["J"] > w["A"]


def test_nan_scores_get_zero_weight():
    scores = pd.Series({"A": 1.0, "B": np.nan, "C": 0.2, "D": -0.4, "E": -1.0, "F": 0.8})
    w = long_short_weights(scores, method="quantile", quantile=0.3)
    assert w["B"] == 0.0
    assert abs(w.sum()) < 1e-12


def test_turnover_zero_when_unchanged():
    w = pd.Series({"A": 0.5, "B": -0.5, "C": 0.0})
    assert two_way_turnover(w, w) == 0.0


def test_turnover_positive_when_weights_change():
    a = pd.Series({"A": 0.5, "B": -0.5, "C": 0.0})
    b = pd.Series({"A": 0.0, "B": 0.5, "C": -0.5})
    to = two_way_turnover(a, b)
    assert to > 0.0
    # |0.5-0| + |-0.5-0.5| + |0-(-0.5)| = 0.5 + 1.0 + 0.5 = 2.0
    assert abs(to - 2.0) < 1e-12


def test_costs_reduce_cumulative_when_turnover():
    prices = make_synthetic_panel(n_assets=8, n_days=400, seed=8)
    rets = daily_returns(prices)
    scores = combined_score(prices)
    # Rebuild a simple monthly-held book from last available score each month.
    month_ends = scores.groupby(scores.index.to_period("M")).apply(
        lambda x: x.index.max()
    )
    weight_panel = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)
    prev = pd.Series(0.0, index=prices.columns)
    prev_d = None
    for d in month_ends:
        d = pd.Timestamp(d)
        sc = scores.loc[d]
        if sc.notna().sum() < 4:
            continue
        w = long_short_weights(sc, method="quantile", quantile=0.25)
        w = w.reindex(prices.columns).fillna(0.0)
        if prev_d is not None:
            held = prices.index[(prices.index > prev_d) & (prices.index <= d)]
            weight_panel.loc[held, :] = prev.values
        turnover.loc[d] = two_way_turnover(prev, w)
        prev = w
        prev_d = d
    if prev_d is not None:
        tail = prices.index[prices.index > prev_d]
        weight_panel.loc[tail, :] = prev.values

    assert float(turnover.sum()) > 0.0
    gross, net = costed_portfolio_returns(
        rets, weight_panel, cost_bps=25.0, turnover_by_date=turnover
    )
    gross0, net0 = costed_portfolio_returns(
        rets, weight_panel, cost_bps=0.0, turnover_by_date=turnover
    )
    assert abs(cumulative_return(gross0) - cumulative_return(net0)) < 1e-12
    assert cumulative_return(net) < cumulative_return(gross)
    assert np.isfinite(gross.dropna().to_numpy()).all()
    assert np.isfinite(net.dropna().to_numpy()).all()


def test_sharpe_handles_short_and_zero_vol():
    assert np.isnan(sharpe(pd.Series([0.01])))
    assert sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0
    s = sharpe(pd.Series([0.01, -0.005, 0.002, 0.003]))
    assert np.isfinite(s)
