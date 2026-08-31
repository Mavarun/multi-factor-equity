#!/usr/bin/env python3
"""Run the momentum + low-vol slice on a mega-cap panel (or synthetic).

Hypothesis (research demo, not live PnL):
1. Cross-sectional 12-1 momentum and low-volatility scores predict
   next-month returns after simple market neutralization.
2. Ignoring turnover and per-trade costs inflates long-short Sharpe;
   net of costs the edge shrinks or vanishes.
3. Walk-forward OOS rank IC is weaker than pooled in-sample IC;
   report both honestly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multi_factor_equity.data import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_UNIVERSE,
    load_demo_prices,
)
from multi_factor_equity.walkforward import walk_forward


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--neutralize", action="store_true", default=True)
    parser.add_argument("--no-neutralize", action="store_false", dest="neutralize")
    args = parser.parse_args()

    cache = ROOT / "data" / "yf_cache.csv"
    if args.synthetic:
        from multi_factor_equity.data import make_synthetic_panel

        n_days = 1512
        prices = make_synthetic_panel(
            n_assets=len(DEFAULT_UNIVERSE),
            n_days=n_days,
            seed=42,
            start=args.start,
            names=list(DEFAULT_UNIVERSE),
        )
        source = "synthetic"
        print(
            "WARNING: --synthetic set; using seeded synthetic panel. "
            "Not market data."
        )
    else:
        prices, source = load_demo_prices(
            tickers=list(DEFAULT_UNIVERSE),
            start=args.start,
            end=args.end,
            cache_path=cache,
            allow_synthetic=True,
        )

    result = walk_forward(
        prices,
        cost_bps=args.cost_bps,
        method="quantile",
        quantile=args.quantile,
        neutralize_returns=args.neutralize,
    )
    m = result.metrics
    m["source"] = source
    m["n_names"] = int(prices.shape[1])
    m["n_days"] = int(prices.shape[0])
    m["price_start"] = str(prices.index[0].date())
    m["price_end"] = str(prices.index[-1].date())
    m["names"] = list(prices.columns)

    print(json.dumps(m, indent=2, default=str))
    print()
    print(
        f"{'metric':<22} {'value':>10}    source={source}  "
        f"names={m['n_names']}  rebalances={m['n_rebalances']}"
    )
    print("-" * 56)
    rows = [
        ("gross Sharpe (OOS)", m["gross_sharpe"]),
        ("net Sharpe (OOS)", m["net_sharpe"]),
        ("mean turnover (OOS)", m["mean_turnover"]),
        ("mean |IC| (OOS)", m["mean_abs_ic_oos"]),
        ("pooled IC (IS)", m["ic_is_pooled"]),
        ("pooled IC (OOS)", m["ic_oos_pooled"]),
        ("cum gross (OOS)", m["cum_gross_oos"]),
        ("cum net (OOS)", m["cum_net_oos"]),
    ]
    for label, val in rows:
        print(f"{label:<22} {_fmt(val):>10}")
    print()
    print(
        "Note: OOS Sharpe/IC are the second half of month-end rebalances "
        "under trailing-only scores. Costs = "
        f"{args.cost_bps:.1f} bps per unit two-way turnover. "
        "Not live PnL, not a claim this beats the market."
    )


if __name__ == "__main__":
    main()
