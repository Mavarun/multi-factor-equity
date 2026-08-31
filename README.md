# multi-factor-equity

Research slice: **cross-sectional 12–1 momentum + low realized vol**, combined after z-scoring, held in a **month-end long-short** with **turnover costs** and a **walk-forward OOS** split.

Public Yahoo mega-caps only (or a seeded synthetic panel if the download fails). This is a method check. **Not live PnL, not production alpha, not a claim this beats the market.**

## Today's hypothesis

1. Cross-sectional 12–1 momentum and low-volatility scores predict next-month returns after simple market neutralization.
2. Ignoring turnover and per-trade costs inflates long-short Sharpe; net of costs the edge shrinks or vanishes.
3. Walk-forward OOS rank IC is weaker than pooled in-sample IC; report both honestly.

## Method

- **Momentum (12–1)**: at date *t*, `P[t-21] / P[t-252] - 1` (skip the most recent ~1 month).
- **Low-vol**: trailing 63-day realized vol of daily returns; the score is **minus** the cross-sectional z-score (quiet names rank high).
- **Combination**: `0.5 * z(mom) + 0.5 * z(-vol)`. Optional equal-weight residualization `r_i - mean_j r_j` before both legs (demo default: on).
- **Portfolio**: top/bottom 20% equal-weight, dollar-neutral, gross exposure 1.0. Rank-weighted is implemented as an alternative.
- **Rebalance**: last session of each calendar month. Scores at *t* use `prices.loc[:t]` only. The book decided at close *t* earns daily returns from *t+1* through the next month-end.
- **Costs**: `cost_bps / 1e4 * sum(|Δw|)` charged on the rebalance date (two-way turnover). Demo uses **10 bps** per unit turnover — a round-trip of the full 1.0-gross book is 20 bps.
- **IC**: Spearman rank correlation of the score at *t* vs the holding-period return `P[next]/P[t] - 1`.
- **IS / OOS**: first half of month-end rebalances vs the second half. Factor construction is trailing on both halves; the split is an evaluation window, not a fitted parameter.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_momentum_vol_slice.py
```

`pytest` is synthetic-only (no network). The demo tries yfinance for

`AAPL MSFT GOOGL AMZN META NVDA JPM XOM JNJ PG UNH HD V MA COST`

over 2018-01-02 … 2026-07-31, caches `data/yf_cache.csv` (gitignored), and **falls back to a seeded synthetic panel with a printed warning** if the download fails.

## Results (this slice)

Universe: 15 US mega/large caps, Adj Close via yfinance, 2018-01-02 → 2026-07-30 (2,155 sessions). 90 month-end rebalances after the 252-day lookback. IS = 2019-01 … 2022-09 (45 months). OOS = 2022-10 … 2026-06 (45 months). Neutralized returns, 20% quantile, 10 bps / unit two-way turnover.

| metric | value |
| --- | ---: |
| OOS gross Sharpe (daily, ann.) | **−0.48** |
| OOS net Sharpe | **−0.53** |
| mean two-way turnover (OOS) | 0.56 |
| mean \|IC\| (OOS months) | 0.26 |
| pooled Spearman IC (IS) | **+0.035** |
| pooled Spearman IC (OOS) | **−0.085** |
| OOS cumulative gross / net | −22.4% / −24.4% |

Hypothesis (1) is **not supported** on this mega-cap panel: the pooled IS IC is only barely positive and the OOS IC flips sign. Mega-cap 2022–2026 was a few-winner tape (NVDA and friends); a 20% long/short of 15 names is mostly trading the same crowded factor. That is a real negative for *this* universe, not a licence to hide the number.

Hypothesis (2) holds in the accounting sense: 10 bps of cost on ~0.56 two-way monthly turnover takes OOS Sharpe from −0.48 to −0.53 and cumulative from −22.4% to −24.4%. Gross already had no edge; costs made a bad book worse. Reporting only gross Sharpe would have been the dishonest version of the same story.

Hypothesis (3) holds: pooled IS IC (+0.035) is weaker out of sample (−0.085). Mean monthly \|IC\| is also a bit lower OOS (0.26 vs 0.29 IS). Full-sample pooled IC is −0.021 — the number you get if you refuse to split.

## Assumptions and limits

- Close-to-close, Adj Close, 15 survivors. No borrow, no locates, no lots, no impact beyond a flat bps×turnover term.
- Equal-weight residualization is beta = 1, not a trailing beta. Cross-sectional z-score uses sklearn `StandardScaler` (population std) on that date's finite names only.
- 10 bps per unit two-way turnover is a guess, not a tape. One-way ~28 bps/month at the observed OOS turnover.
- First/last month-end after lookback define the sample; the last month is used only as a forward-return endpoint, not as a decision date.
- **Do not read these Sharpes as live PnL.** A 15-name mega-cap long-short is a teaching panel, not a product.

## Layout

```
src/multi_factor_equity/
  factors.py       # 12-1 mom, realized vol, CS z-score, residualize
  portfolio.py     # quantile / rank LS weights, turnover, costed returns
  walkforward.py   # month-end trailing-only WF, Spearman IC, IS/OOS split
  data.py          # synthetic panel + yfinance Adj Close loader
scripts/run_momentum_vol_slice.py
tests/             # synthetic only; leakage guard with a planted future spike
```

## Next slices (not done)

- Broader universe (S&P 500-ish) so 20% quantiles are not three names
- Value / quality as a third z-scored leg, plus explicit factor-correlation report
- Trailing-beta residualization and a sector-neutral variant
- Cost grid and turnover-aware shrinkage of the rank book
