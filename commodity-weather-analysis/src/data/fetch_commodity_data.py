"""
fetch_commodity_data.py
-----------------------
Downloads historical daily price data for all eight commodities using yfinance.

Commodity pairs studied:
    Agricultural  : Corn/Oats, Wheat/Soybean, Coffee/Cocoa
    Non-Agricultural: Gold/Silver

Improvement over original paper:
    - Uses yfinance instead of manual CSV downloads from macrotrends/quandl
    - Data extended to 2024
    - Consistent date alignment across all commodities via inner/outer join options
    - Missing value interpolation with forward-fill fallback
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker mapping  (yfinance symbols)
# ---------------------------------------------------------------------------
COMMODITY_TICKERS = {
    "Corn":    "ZC=F",
    "Oats":    "ZO=F",
    "Wheat":   "ZW=F",
    "Soybean": "ZS=F",
    "Coffee":  "KC=F",
    "Cocoa":   "CC=F",
    "Gold":    "GC=F",
    "Silver":  "SI=F",
}

COMMODITY_PAIRS = [
    ("Corn",    "Oats"),
    ("Wheat",   "Soybean"),
    ("Coffee",  "Cocoa"),
    ("Gold",    "Silver"),
]

DEFAULT_START = "1973-01-01"
DEFAULT_END   = "2024-12-31"
RAW_DATA_DIR  = Path(__file__).parents[2] / "data" / "raw"
PROC_DATA_DIR = Path(__file__).parents[2] / "data" / "processed"


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_commodity(
    name: str,
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.Series:
    """Download adjusted-close price for a single commodity.

    Parameters
    ----------
    name   : Human-readable commodity name
    ticker : yfinance ticker symbol
    start  : Start date (YYYY-MM-DD)
    end    : End date   (YYYY-MM-DD)

    Returns
    -------
    pd.Series  — daily close prices indexed by date
    """
    logger.info(f"Fetching {name} ({ticker})  {start} → {end}")
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        logger.warning(f"  No data returned for {name}. Check ticker or date range.")
        return pd.Series(dtype=float, name=name)
    series = raw["Close"].squeeze().rename(name)
    series.index = pd.to_datetime(series.index)
    return series


def fetch_all_commodities(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    join: str = "outer",
    save: bool = True,
) -> pd.DataFrame:
    """Download all eight commodities and return a combined DataFrame.

    Parameters
    ----------
    start : Start date
    end   : End date
    join  : 'outer' (keep all dates) or 'inner' (only common trading days)
    save  : If True, saves to data/raw/commodity_prices.csv

    Returns
    -------
    pd.DataFrame  — columns = commodity names, index = date
    """
    series_list = [
        fetch_commodity(name, ticker, start, end)
        for name, ticker in COMMODITY_TICKERS.items()
    ]
    df = pd.concat(series_list, axis=1, join=join)
    df.index.name = "Date"
    df = _clean_prices(df)

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = RAW_DATA_DIR / "commodity_prices.csv"
        df.to_csv(out)
        logger.info(f"Saved raw prices → {out}")

    return df


# ---------------------------------------------------------------------------
# Cleaning & preprocessing
# ---------------------------------------------------------------------------

def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values; forward/back fill at boundaries."""
    df = df.copy()
    # Remove rows where all values are NaN (weekends/holidays)
    df.dropna(how="all", inplace=True)
    # Linear interpolation for isolated missing days
    df.interpolate(method="time", inplace=True)
    # Fill any remaining edge NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def compute_returns(
    prices: pd.DataFrame,
    freq: str = "D",
) -> pd.DataFrame:
    """Compute percentage returns at daily ('D'), weekly ('W'), or monthly ('ME') frequency.

    Parameters
    ----------
    prices : DataFrame of prices (columns = commodities)
    freq   : Resampling frequency — 'D', 'W', or 'ME'

    Returns
    -------
    pd.DataFrame of returns
    """
    if freq != "D":
        prices = prices.resample(freq).last()
    returns = prices.pct_change().dropna()
    return returns


def compute_spread(
    prices: pd.DataFrame,
    pair: tuple,
) -> pd.Series:
    """Compute price spread for a commodity pair.

    Parameters
    ----------
    prices : DataFrame of prices
    pair   : Tuple of two column names, e.g. ('Gold', 'Silver')
    """
    a, b = pair
    spread = prices[a] - prices[b]
    spread.name = f"{a}_minus_{b}"
    return spread


def compute_ratio(
    prices: pd.DataFrame,
    pair: tuple,
) -> pd.Series:
    """Compute price ratio for a commodity pair."""
    a, b = pair
    ratio = prices[a] / prices[b]
    ratio.name = f"{a}_over_{b}"
    return ratio


def build_processed_dataset(
    prices: pd.DataFrame,
    save: bool = True,
) -> dict:
    """Build a complete processed dataset dictionary with prices,
    returns at all frequencies, spreads, and ratios.

    Returns
    -------
    dict with keys: 'prices', 'daily_returns', 'weekly_returns',
                    'monthly_returns', 'spreads', 'ratios'
    """
    dataset = {
        "prices":          prices,
        "daily_returns":   compute_returns(prices, "D"),
        "weekly_returns":  compute_returns(prices, "W"),
        "monthly_returns": compute_returns(prices, "ME"),
        "spreads":         {f"{a}_{b}": compute_spread(prices, (a, b))
                            for a, b in COMMODITY_PAIRS},
        "ratios":          {f"{a}_{b}": compute_ratio(prices, (a, b))
                            for a, b in COMMODITY_PAIRS},
    }

    if save:
        PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for key in ["prices", "daily_returns", "weekly_returns", "monthly_returns"]:
            path = PROC_DATA_DIR / f"{key}.csv"
            dataset[key].to_csv(path)
            logger.info(f"Saved {key} → {path}")

    return dataset


# ---------------------------------------------------------------------------
# Convenience loader (skip download if file exists)
# ---------------------------------------------------------------------------

def load_or_fetch(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Load prices from disk if available, otherwise fetch from yfinance."""
    cached = RAW_DATA_DIR / "commodity_prices.csv"
    if cached.exists():
        logger.info(f"Loading cached prices from {cached}")
        df = pd.read_csv(cached, index_col="Date", parse_dates=True)
        return df
    return fetch_all_commodities(start=start, end=end, save=True)


# ---------------------------------------------------------------------------
# Descriptive statistics (replicates original paper Table 4.1)
# ---------------------------------------------------------------------------

def descriptive_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive statistics table matching original paper Table 4.1."""
    stats = prices.describe().T
    stats["skewness"] = prices.skew()
    stats["kurtosis"] = prices.kurt()
    return stats.round(4)


if __name__ == "__main__":
    prices = fetch_all_commodities()
    print(descriptive_stats(prices))
