import numpy as np
import pandas as pd

from multi_factor_equity.data import make_synthetic_panel
from multi_factor_equity.factors import score_asof
from multi_factor_equity.portfolio import cumulative_return
from multi_factor_equity.walkforward import month_end_dates, walk_forward


def test_month_end_dates_are_last_session():
    idx = pd.bdate_range("2020-01-02", periods=60)
    ends = month_end_dates(idx)
    assert len(ends) >= 2
    for d in ends:
        month = d.to_period("M")
        in_month = idx[idx.to_period("M") == month]
        assert d == in_month.max()


def test_walkforward_no_nan_blowups():
    prices = make_synthetic_panel(n_assets=10, n_days=800, seed=11)
    result = walk_forward(prices, cost_bps=10.0, quantile=0.2)
    g = result.daily_gross.replace([np.inf, -np.inf], np.nan)
    n = result.daily_net.replace([np.inf, -np.inf], np.nan)
    # After the first rebalance the book is live; allow the leading zeros.
    live = g.loc[result.rebalances[0].date :]
    assert live.isna().sum() == 0
    assert n.loc[result.rebalances[0].date :].isna().sum() == 0
    assert np.isfinite(result.metrics["gross_sharpe"]) or np.isnan(
        result.metrics["gross_sharpe"]
    )
    assert result.metrics["n_rebalances"] >= 8
    for rec in result.rebalances:
        assert rec.scores.notna().sum() >= 4
        assert abs(rec.weights.sum()) < 1e-8


def test_walkforward_costs_reduce_net_vs_gross():
    prices = make_synthetic_panel(n_assets=10, n_days=800, seed=12)
    result = walk_forward(prices, cost_bps=20.0, quantile=0.2)
    mean_to = result.metrics["mean_turnover_full"]
    assert mean_to > 0.0
    cum_g = cumulative_return(result.daily_gross)
    cum_n = cumulative_return(result.daily_net)
    assert cum_n < cum_g
    # Full-sample net Sharpe should not exceed gross when costs bite
    # (can fail in pathological negative-gross samples; check cum only).
    assert result.metrics["cum_net_full"] < result.metrics["cum_gross_full"]


def test_future_spike_does_not_affect_earlier_scores():
    """Leakage guard: a planted future spike must not move earlier scores.

    walk_forward slices prices.loc[:t] at each rebalance. If someone later
    z-scores factors over the full time axis, this test fails.
    """
    prices = make_synthetic_panel(n_assets=10, n_days=900, seed=13)
    spiked = prices.copy()
    spike_i = 560
    spike_date = spiked.index[spike_i]
    spiked.iloc[spike_i : spike_i + 4, 0] *= 8.0

    clean = walk_forward(prices, cost_bps=10.0, quantile=0.2)
    dirty = walk_forward(spiked, cost_bps=10.0, quantile=0.2)

    assert [r.date for r in clean.rebalances] == [r.date for r in dirty.rebalances]

    earlier = 0
    later_changed = 0
    for rec_c, rec_s in zip(clean.rebalances, dirty.rebalances):
        if rec_c.date < spike_date:
            pd.testing.assert_series_equal(
                rec_c.scores.sort_index(),
                rec_s.scores.sort_index(),
                check_names=False,
                atol=1e-12,
                rtol=0.0,
            )
            earlier += 1
        elif rec_c.date > spike_date + pd.Timedelta(days=25):
            if not np.allclose(
                rec_c.scores.fillna(0.0).sort_index(),
                rec_s.scores.fillna(0.0).sort_index(),
                atol=1e-10,
            ):
                later_changed += 1
    assert earlier >= 3
    assert later_changed >= 1


def test_score_asof_matches_walkforward_and_ignores_future_rows():
    prices = make_synthetic_panel(n_assets=8, n_days=700, seed=14)
    result = walk_forward(prices, cost_bps=5.0, quantile=0.25)
    rec = result.rebalances[3]
    sliced = score_asof(prices.loc[: rec.date])
    pd.testing.assert_series_equal(
        rec.scores.sort_index(), sliced.sort_index(), check_names=False
    )
    # Extra future rows after rec.date must not change score_asof if we
    # still pass only the slice.
    future_only = prices.copy()
    future_only.iloc[-5:, :] *= 3.0
    sliced_from_spiked_frame = score_asof(future_only.loc[: rec.date])
    pd.testing.assert_series_equal(
        sliced.sort_index(),
        sliced_from_spiked_frame.sort_index(),
        check_names=False,
    )


def test_is_oos_split_covers_all_rebalances():
    prices = make_synthetic_panel(n_assets=10, n_days=900, seed=15)
    result = walk_forward(prices, cost_bps=10.0, is_frac=0.5)
    assert result.metrics["n_is"] + result.metrics["n_oos"] == result.metrics["n_rebalances"]
    assert result.metrics["n_oos"] >= 1
    assert result.metrics["n_is"] >= 1
