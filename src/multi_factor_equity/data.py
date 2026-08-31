"""Price panel loaders: seeded synthetic (tests) and yfinance (demo)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "UNH",
    "HD",
    "V",
    "MA",
    "COST",
]

DEFAULT_START = "2018-01-02"
DEFAULT_END = "2026-07-31"


def make_synthetic_panel(
    n_assets: int = 12,
    n_days: int = 1512,
    seed: int = 42,
    start: str = "2018-01-02",
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Seeded daily close panel with a market factor, vol dispersion, and AR drifts.

    Not market data. Used by tests and as an offline demo fallback.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    if names is None:
        names = [f"A{i:02d}" for i in range(n_assets)]
    else:
        names = list(names)
        n_assets = len(names)

    # Heterogeneous vols: first names quieter (low-vol leg has something to rank).
    daily_vol = np.linspace(0.10, 0.40, n_assets) / np.sqrt(252.0)
    mkt = rng.normal(0.07 / 252.0, 0.15 / np.sqrt(252.0), size=n_days)

    idio = np.zeros((n_days, n_assets))
    drift = np.zeros(n_assets)
    for t in range(n_days):
        drift = 0.97 * drift + rng.normal(0.0, 0.0005, size=n_assets)
        shock = rng.normal(0.0, 1.0, size=n_assets) * daily_vol
        idio[t] = drift + shock

    log_rets = 0.55 * mkt[:, None] + idio
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=names)


def _extract_adj_close(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Adj Close" in set(level0):
            out = raw["Adj Close"].copy()
        elif "Close" in set(level0):
            out = raw["Close"].copy()
        else:
            raise ValueError(f"unexpected yfinance columns: {raw.columns}")
    elif "Adj Close" in raw.columns:
        out = raw[["Adj Close"]].copy()
        out.columns = ["PX"]
    else:
        out = raw.copy()
    if isinstance(out, pd.Series):
        out = out.to_frame()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out.astype(float)
    return out


def load_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_path: str | Path | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download Adj Close for a small liquid US universe. Optional CSV cache."""
    import yfinance as yf

    tickers = list(tickers or DEFAULT_UNIVERSE)
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists() and not force_download:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        return cached.loc[start:end]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned an empty frame")
    prices = _extract_adj_close(raw)
    # Keep requested order; drop all-NaN names.
    keep = [t for t in tickers if t in prices.columns]
    prices = prices.loc[:, keep]
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill(limit=3)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return prices


def load_demo_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_path: str | Path | None = None,
    allow_synthetic: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Try yfinance (and cache); fall back to synthetic with a warning string.

    Returns (prices, source) where source is "yfinance" or "synthetic".
    """
    tickers = list(tickers or DEFAULT_UNIVERSE)
    try:
        prices = load_prices(
            tickers=tickers,
            start=start,
            end=end,
            cache_path=cache_path,
        )
        if prices.shape[1] < 8 or prices.shape[0] < 400:
            raise RuntimeError(
                f"downloaded panel too small: {prices.shape}"
            )
        return prices, "yfinance"
    except Exception as exc:
        if not allow_synthetic:
            raise
        warning = (
            f"WARNING: yfinance load failed ({exc!r}); "
            "using seeded synthetic panel. Not market data."
        )
        print(warning)
        n_days = len(pd.bdate_range(start, end))
        syn = make_synthetic_panel(
            n_assets=len(tickers),
            n_days=max(n_days, 800),
            seed=42,
            start=start,
            names=tickers,
        )
        # Trim to requested calendar end if the bdate range overshot.
        syn = syn.loc[:end]
        return syn, "synthetic"
