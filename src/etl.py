"""
Station 1 - ETL: load, clean, and quality-check the data.

Design choices worth knowing when you read this:
- We CAP the sample at 2023-12-31 (the 10 stray crypto rows dated 2024-01-01).
- Prices are de-duplicated on (ticker, date). News is de-duplicated on
  (ticker, date, title), because many rows per ticker-date is normal for news and
  a ticker-date-only check would wrongly flag everything (DATA_GUIDE).
- Outliers are FLAGGED and KEPT, never deleted (they are real market events).
- News dates are timezone-aware UTC; price dates are naive. We normalise the news
  dates to naive day-resolution timestamps before anything is merged.

Each loader returns (clean_dataframe, meta_dict) so run_part_a.py can build the
integrity summary from what was actually found, not from hard-coded numbers.
"""
from __future__ import annotations

import pandas as pd

from src import data_access
from src import config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cap_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows on or before SAMPLE_END (removes the stray 2024 crypto rows)."""
    end = pd.Timestamp(config.SAMPLE_END)
    return df.loc[df["date"] <= end].copy()


def _to_naive_day(s: pd.Series) -> pd.Series:
    """Normalise a datetime Series to tz-naive, day-resolution timestamps.

    News dates are tz-aware UTC (datetime64[us, UTC]); price dates are tz-naive.
    We convert to UTC, drop the tz, then floor to midnight so a headline maps to a
    calendar day the price calendar also understands.
    """
    s = pd.to_datetime(s)
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    # Force nanosecond resolution. Real Parquet stores dates at mixed resolutions
    # (news in [us], prices in [s]); merge_asof requires both keys identical.
    return s.dt.normalize().astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# loaders + cleaning
# ---------------------------------------------------------------------------
def load_clean_equities():
    """Load equity prices, cap the window, and de-duplicate on (ticker, date)."""
    df = data_access.load_equity_prices()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    df = _cap_sample(df)

    n_before = len(df)
    dup_mask = df.duplicated(subset=["ticker", "date"], keep="first")
    n_dupes = int(dup_mask.sum())
    df = df.loc[~dup_mask].sort_values(["ticker", "date"]).reset_index(drop=True)

    meta = {
        "rows_clean": len(df),
        "duplicate_ticker_date_removed": n_dupes,
        "rows_before_dedup": n_before,
        "tickers": df["ticker"].nunique(),
        "sectors": df["sector"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
    }
    return df, meta


def load_clean_crypto():
    """Load crypto prices (365-day calendar), cap the window, de-duplicate."""
    df = data_access.load_crypto_prices()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")

    n_after_2023 = int((df["date"] > pd.Timestamp(config.SAMPLE_END)).sum())
    df = _cap_sample(df)

    n_before = len(df)
    dup_mask = df.duplicated(subset=["ticker", "date"], keep="first")
    n_dupes = int(dup_mask.sum())
    df = df.loc[~dup_mask].sort_values(["ticker", "date"]).reset_index(drop=True)

    meta = {
        "rows_clean": len(df),
        "rows_dated_after_2023_removed": n_after_2023,
        "duplicate_ticker_date_removed": n_dupes,
        "rows_before_dedup": n_before,
        "tickers": df["ticker"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
    }
    return df, meta


def load_clean_news():
    """Load headlines, cap the window, normalise timezone, de-dup on ticker+date+title."""
    df = data_access.load_news_headlines()
    df["date"] = _to_naive_day(df["date"])
    df = _cap_sample(df)

    n_before = len(df)
    dup_mask = df.duplicated(subset=["ticker", "date", "title"], keep="first")
    n_dupes = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)

    meta = {
        "rows_clean": len(df),
        "exact_duplicate_headlines_removed": n_dupes,
        "rows_before_dedup": n_before,
        "tickers": df["ticker"].nunique(),
        "sectors": df["sector"].nunique(),
        "blank_publisher_share": round(float(df["publisher"].isna().mean()
                                             + (df["publisher"] == "").mean()), 4),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
    }
    return df, meta


# ---------------------------------------------------------------------------
# integrity audits
# ---------------------------------------------------------------------------
def equity_trading_calendar(equities: pd.DataFrame) -> pd.DatetimeIndex:
    """The equity trading calendar = the sorted unique equity dates."""
    return pd.DatetimeIndex(sorted(equities["date"].unique()))


def missing_date_audit(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-ticker count of calendar days missing within each ticker's own coverage.

    For each ticker we take the reference `calendar` restricted to that ticker's
    [first, last] observed date, and count how many of those days it is missing.
    """
    rows = []
    cal = pd.DatetimeIndex(calendar)
    for tkr, g in prices.groupby("ticker"):
        present = pd.DatetimeIndex(g["date"].unique())
        lo, hi = present.min(), present.max()
        expected = cal[(cal >= lo) & (cal <= hi)]
        missing = expected.difference(present)
        rows.append({"ticker": tkr, "first": lo, "last": hi,
                     "expected_days": len(expected), "present_days": len(present),
                     "missing_days": len(missing)})
    out = pd.DataFrame(rows).sort_values("missing_days", ascending=False)
    return out.reset_index(drop=True)


def crypto_calendar_gaps(crypto: pd.DataFrame) -> pd.DataFrame:
    """Per-coin count of calendar-day gaps (crypto should be a full 365-day series)."""
    rows = []
    for tkr, g in crypto.groupby("ticker"):
        present = pd.DatetimeIndex(g["date"].unique())
        full = pd.date_range(present.min(), present.max(), freq="D")
        rows.append({"ticker": tkr, "first": present.min(), "last": present.max(),
                     "expected_days": len(full), "present_days": len(present),
                     "missing_days": len(full.difference(present))})
    return pd.DataFrame(rows).sort_values("missing_days", ascending=False).reset_index(drop=True)
