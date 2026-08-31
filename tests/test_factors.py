import numpy as np
import pandas as pd
import pytest

from multi_factor_equity.data import make_synthetic_panel
from multi_factor_equity.factors import (
    combined_score,
    cross_sectional_zscore,
    daily_returns,
    momentum_12_1,
    residualize_equal_weight,
    realized_vol,
    score_asof,
)


def test_synthetic_panel_is_seeded():
    a = make_synthetic_panel(n_assets=8, n_days=100, seed=1)
    b = make_synthetic_panel(n_assets=8, n_days=100, seed=1)
    c = make_synthetic_panel(n_assets=8, n_days=100, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_momentum_warmup_and_no_nan_blowup():
    prices = make_synthetic_panel(n_assets=10, n_days=400, seed=3)
    mom = momentum_12_1(prices, lookback=252, skip=21)
    assert mom.shape == prices.shape
    assert mom.iloc[:252].isna().all().all()
    tail = mom.iloc[252:]
    assert tail.notna().any().any()
    assert np.isfinite(tail.to_numpy()[np.isfinite(tail.to_numpy())]).all()


def test_momentum_positive_on_monotonic_uptrend():
    idx = pd.bdate_range("2020-01-02", periods=300)
    up = pd.Series(np.linspace(100.0, 200.0, 300), index=idx)
    down = pd.Series(np.linspace(200.0, 100.0, 300), index=idx)
    prices = pd.DataFrame({"UP": up, "DOWN": down})
    mom = momentum_12_1(prices, lookback=252, skip=21)
    last = mom.iloc[-1]
    assert last["UP"] > 0
    assert last["DOWN"] < 0


def test_realized_vol_ranks_noisier_name_higher():
    idx = pd.bdate_range("2020-01-02", periods=250)
    rng = np.random.default_rng(0)
    quiet = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 250)))
    noisy = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 250)))
    prices = pd.DataFrame({"Q": quiet, "N": noisy}, index=idx)
    vol = realized_vol(daily_returns(prices), window=63)
    assert vol.iloc[-1]["N"] > vol.iloc[-1]["Q"]


def test_zscore_mean_near_zero_std_near_one():
    prices = make_synthetic_panel(n_assets=12, n_days=300, seed=4)
    mom = momentum_12_1(prices, lookback=120, skip=10)
    z = cross_sectional_zscore(mom)
    row = z.iloc[-1].dropna()
    assert abs(row.mean()) < 1e-8
    assert abs(row.std(ddof=0) - 1.0) < 1e-8


def test_zscore_handles_nan_without_poisoning_row():
    idx = pd.bdate_range("2020-01-02", periods=5)
    panel = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, 5.0],
            "B": [2.0, 3.0, 4.0, 5.0, 6.0],
            "C": [3.0, 4.0, 5.0, np.nan, 7.0],
        },
        index=idx,
    )
    z = cross_sectional_zscore(panel)
    assert np.isfinite(z.loc[idx[0]].to_numpy()).all()
    assert np.isnan(z.loc[idx[2], "A"])
    assert np.isfinite(z.loc[idx[2], ["B", "C"]].to_numpy()).all()


def test_residualize_cross_sectional_mean_zero():
    prices = make_synthetic_panel(n_assets=10, n_days=200, seed=5)
    rets = daily_returns(prices).iloc[1:]
    resid = residualize_equal_weight(rets)
    cs_mean = resid.mean(axis=1)
    assert np.nanmax(np.abs(cs_mean.to_numpy())) < 1e-10


def test_combined_score_finite_after_warmup():
    prices = make_synthetic_panel(n_assets=10, n_days=400, seed=6)
    scores = combined_score(prices)
    tail = scores.iloc[252:]
    finite_frac = np.isfinite(tail.to_numpy()).mean()
    assert finite_frac > 0.9
    last = score_asof(prices)
    assert last.notna().sum() >= 8


def test_neutralized_scores_run():
    prices = make_synthetic_panel(n_assets=8, n_days=400, seed=7)
    raw = combined_score(prices, neutralize_returns=False)
    neu = combined_score(prices, neutralize_returns=True)
    assert raw.shape == neu.shape
    assert np.isfinite(neu.iloc[-1].to_numpy()).sum() >= 6
