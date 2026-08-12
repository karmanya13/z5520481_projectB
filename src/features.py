"""
Station 2 - Feature engineering and text assembly.

Returns:
- Daily simple returns per ticker from adjClose, computed WITHIN each panel first.
- The combined panel left-merges crypto returns onto the equity trading calendar
  (this intentionally drops weekend-only crypto moves, which an equity-day fund
  could not trade on). We never merge price levels across calendars and difference
  afterwards, because that would manufacture spurious returns.

Text (Part A is ASSEMBLY ONLY):
- Every headline is mapped to its equity trading day (the same day if it trades,
  otherwise the next trading day) via a forward as-of merge.
- Raw headline text is kept intact (VADER relies on stopwords and casing, so we do
  not strip them from the stored panel). Scoring the text into a sentiment index,
  and lagging that signal, is the Station 3 model in Part B - not here.
- Counting sentiment-bearing VOCABULARY is descriptive and allowed in Part A; it is
  a word count, not a sentiment score.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from src import config

_TOKEN_RE = re.compile(r"[a-z][a-z']+")


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose",
                  asset_class: str | None = None) -> pd.DataFrame:
    """Simple daily returns per ticker (long format: ticker, date, ret).

    fill_method=None so pct_change does not silently forward-fill across gaps.
    The first observation per ticker is NaN by construction and is dropped.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")[price_col].pct_change(fill_method=None)
    out = df.loc[df["ret"].notna(), ["ticker", "date", "ret"]].copy()
    if asset_class is not None:
        out["asset_class"] = asset_class
    return out.reset_index(drop=True)


def combined_returns_panel(equity_ret: pd.DataFrame, crypto_ret: pd.DataFrame,
                           equity_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Combined long panel on the EQUITY calendar.

    Equity returns pass through as-is. Crypto returns are left-merged onto the
    equity trading days, so only crypto moves that fall on an equity trading day
    survive (weekend-only crypto moves are dropped, by design).
    """
    eq = equity_ret.copy()
    eq["asset_class"] = "equity"

    cal = pd.DatetimeIndex(equity_calendar)
    cr = crypto_ret[crypto_ret["date"].isin(cal)].copy()
    cr["asset_class"] = "crypto"

    cols = ["date", "ticker", "asset_class", "ret"]
    panel = pd.concat([eq[cols], cr[cols]], ignore_index=True)
    return panel.sort_values(["date", "asset_class", "ticker"]).reset_index(drop=True)


def descriptive_stats_by_class(equity_ret: pd.DataFrame,
                               crypto_ret: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics per asset class on each class's NATIVE calendar.

    Crypto is summarised on its own 365-day series (not the equity-aligned one), so
    the annualisation factor matches the data. Equity uses 252, crypto uses 365.
    """
    def _stats(r: pd.Series, factor: int) -> dict:
        r = r.dropna()
        return {
            "n_obs": int(r.shape[0]),
            "mean_daily": r.mean(),
            "vol_daily": r.std(ddof=1),
            "min": r.min(),
            "max": r.max(),
            "skew": stats.skew(r, bias=False),
            "excess_kurtosis": stats.kurtosis(r, fisher=True, bias=False),
            "ann_return": r.mean() * factor,
            "ann_vol": r.std(ddof=1) * np.sqrt(factor),
        }

    rows = {
        "equity": _stats(equity_ret["ret"], config.EQUITY_TRADING_DAYS),
        "crypto": _stats(crypto_ret["ret"], config.CRYPTO_TRADING_DAYS),
    }
    out = pd.DataFrame(rows).T
    out.index.name = "asset_class"
    return out.reset_index()


def flag_extreme_returns(panel: pd.DataFrame, threshold: float = None,
                         top_n: int = None) -> pd.DataFrame:
    """List the largest-magnitude daily returns per asset class (flagged, kept).

    This backs the outlier row of the integrity summary. Nothing is deleted here;
    it just surfaces the extremes so they can be shown and explained in the report.
    """
    threshold = config.OUTLIER_ABS_RETURN if threshold is None else threshold
    top_n = config.OUTLIER_TOP_N if top_n is None else top_n
    ex = panel.loc[panel["ret"].abs() > threshold].copy()
    ex["abs_ret"] = ex["ret"].abs()
    out = (ex.sort_values("abs_ret", ascending=False)
             .groupby("asset_class", group_keys=False)
             .head(top_n)
             .sort_values(["asset_class", "abs_ret"], ascending=[True, False]))
    return out[["asset_class", "ticker", "date", "ret"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# text assembly
# ---------------------------------------------------------------------------
def assemble_headline_panel(news: pd.DataFrame,
                            equity_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Map every headline to its equity trading day (same day, else next trading day).

    Uses a forward as-of merge: each headline date is matched to the first trading
    day >= that date. Headlines after the last trading day get no match and are
    dropped (reported by run_part_a). Raw title text is preserved.
    """
    cal = pd.DataFrame({"trading_day": pd.DatetimeIndex(sorted(equity_calendar))})
    cal["trading_day"] = cal["trading_day"].astype("datetime64[ns]")
    cal = cal.sort_values("trading_day")

    n = news.sort_values("date").copy()
    # merge_asof needs both keys at the SAME datetime resolution (Parquet mixes them)
    n["date"] = n["date"].astype("datetime64[ns]")
    mapped = pd.merge_asof(n, cal, left_on="date", right_on="trading_day",
                           direction="forward")
    dropped = int(mapped["trading_day"].isna().sum())
    mapped = mapped.loc[mapped["trading_day"].notna()].copy()

    panel = mapped.rename(columns={"date": "headline_date"})[
        ["trading_day", "headline_date", "ticker", "sector", "title", "url", "publisher"]
    ].sort_values(["trading_day", "ticker"]).reset_index(drop=True)
    panel.attrs["headlines_dropped_after_last_trading_day"] = dropped
    return panel


@lru_cache(maxsize=1)
def _vader_lexicon() -> frozenset:
    """VADER's sentiment vocabulary, used ONLY to COUNT sentiment-bearing words.

    Counting vocabulary is a descriptive Part A task. Scoring headlines into a
    sentiment index is Part B and is not done here.
    """
    import nltk
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    return frozenset(sia.lexicon.keys())


def sentiment_word_counts(panel: pd.DataFrame):
    """Descriptive word counts over the assembled headlines.

    Returns three objects:
      summary   - one-row-per-year table: headlines, sentiment-bearing words,
                  share of headlines carrying at least one sentiment word.
      top_terms - the most frequent sentiment-bearing terms overall.
      overall   - a small dict of headline-level totals for the report.
    """
    lex = _vader_lexicon()
    df = panel.copy()

    def _count_sent(title: str) -> int:
        toks = _TOKEN_RE.findall(str(title).lower())
        return sum(1 for t in toks if t in lex)

    df["sent_words"] = df["title"].map(_count_sent)
    df["has_sent"] = df["sent_words"] > 0
    df["year"] = df["trading_day"].dt.year

    summary = (df.groupby("year")
                 .agg(headlines=("title", "size"),
                      sentiment_words=("sent_words", "sum"),
                      share_with_sentiment=("has_sent", "mean"))
                 .reset_index())
    summary["share_with_sentiment"] = summary["share_with_sentiment"].round(4)

    # top sentiment-bearing terms (filtering to the lexicon keeps the chart relevant)
    from collections import Counter
    c = Counter()
    for title in df["title"]:
        for t in _TOKEN_RE.findall(str(title).lower()):
            if t in lex:
                c[t] += 1
    top_terms = pd.DataFrame(c.most_common(20), columns=["term", "count"])

    overall = {
        "headlines_total": int(len(df)),
        "sentiment_words_total": int(df["sent_words"].sum()),
        "share_headlines_with_sentiment": round(float(df["has_sent"].mean()), 4),
        "mean_sentiment_words_per_headline": round(float(df["sent_words"].mean()), 3),
    }
    return summary, top_terms, overall


def headlines_per_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly headline volume, for the 'articles over time' exhibit."""
    df = panel.copy()
    df["month"] = df["trading_day"].dt.to_period("M").dt.to_timestamp()
    return (df.groupby("month").size().rename("headlines").reset_index())
