"""Cross-sectional momentum + low-vol research utilities."""

from multi_factor_equity.data import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_UNIVERSE,
    load_demo_prices,
    load_prices,
    make_synthetic_panel,
)
from multi_factor_equity.factors import (
    combined_score,
    cross_sectional_zscore,
    momentum_12_1,
    residualize_equal_weight,
    realized_vol,
    score_asof,
)
from multi_factor_equity.portfolio import (
    costed_portfolio_returns,
    long_short_weights,
    sharpe,
    two_way_turnover,
)
from multi_factor_equity.walkforward import walk_forward

__all__ = [
    "DEFAULT_END",
    "DEFAULT_START",
    "DEFAULT_UNIVERSE",
    "combined_score",
    "costed_portfolio_returns",
    "cross_sectional_zscore",
    "load_demo_prices",
    "load_prices",
    "long_short_weights",
    "make_synthetic_panel",
    "momentum_12_1",
    "residualize_equal_weight",
    "realized_vol",
    "score_asof",
    "sharpe",
    "two_way_turnover",
    "walk_forward",
]

__version__ = "0.1.0"
