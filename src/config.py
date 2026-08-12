"""
config.py - one place for paths and project constants.

Keeping paths and magic numbers here (not scattered through the code) is what lets
run_part_a.py reproduce the same results from a clean checkout. Every module and
script imports from here.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Folder layout (resolved from this file, so it works regardless of the cwd)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"

for _d in (TABLES_DIR, FIGURES_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample window
# DATA_GUIDE: 10 stray crypto rows are dated 2024-01-01, so cap the sample at the
# end of 2023 to keep a clean four-year window across all three datasets.
# ---------------------------------------------------------------------------
SAMPLE_START = "2020-01-01"
SAMPLE_END = "2023-12-31"

# ---------------------------------------------------------------------------
# Annualisation factors (PROJECT_BRIEF Section 8; DATA_GUIDE "Known traps")
# Equities trade ~252 days a year, crypto ~365. Anything measured on its NATIVE
# calendar annualises with its own factor. The combined panel lives on the
# equity calendar, so it uses 252.
# ---------------------------------------------------------------------------
EQUITY_TRADING_DAYS = 252
CRYPTO_TRADING_DAYS = 365

# ---------------------------------------------------------------------------
# Outlier / extreme-return screen.
# We do NOT delete outliers: the brief is explicit that the extremes here are
# real market events (COVID crash, single-stock shocks, crypto swings). This
# threshold only FLAGS them so they can be listed and explained in the report.
# ---------------------------------------------------------------------------
OUTLIER_ABS_RETURN = 0.20   # daily move flagged as "extreme" when |r| > 20%
OUTLIER_TOP_N = 15          # how many extreme rows to list per asset class

# ---------------------------------------------------------------------------
# Sample assets for the cumulative-return figure.
# These are REAL tickers from context/DATA_GUIDE.md, one per a few sectors plus
# two coins. Change them to whatever you want to showcase once you see the data.
# ---------------------------------------------------------------------------
SAMPLE_EQUITIES = ["NVDA", "GS", "XOM", "DIS", "MRK"]   # Tech, Financials, Energy, Consumer, Healthcare
SAMPLE_CRYPTO = ["BTC-USD", "ETH-USD"]

# ---------------------------------------------------------------------------
# A small, consistent figure style. This is deliberately plain and well-labelled
# (the baseline the rubric asks for). Designing your OWN colour/type system is
# one of the innovation routes for the "distinctive visual quality" top band.
# ---------------------------------------------------------------------------
FIG_DPI = 150
PALETTE = ["#2b3a67", "#e08e45", "#4c956c", "#a4243b", "#7768ae", "#118ab2"]
