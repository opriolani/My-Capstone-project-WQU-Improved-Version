"""
preprocessing.py
================
Cleans raw price and weather data, computes returns (daily/weekly/monthly),
merges commodity prices with weather composites, and saves processed datasets.

Usage:
    python src/preprocessing.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR   = ROOT / CONFIG["paths"]["raw_data"]
PROC_DIR  = ROOT / CONFIG["paths"]["processed_data"]
PROC_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES = list(CONFIG["weather_stations"].keys())


# ══════════════════════════════════════════════════════════════════════════════
# PRICE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class PricePreprocessor:
    """Loads raw price CSVs, cleans, resamples, and computes returns."""

    def __init__(self, raw_dir: Path, proc_dir: Path):
        self.raw_dir  = raw_dir / "prices"
        self.proc_dir = proc_dir

    def process_all(self) -> dict[str, pd.DataFrame]:
        processed = {}
        for commodity in tqdm(COMMODITIES, desc="Processing prices"):
            raw_path = self.raw_dir / f"{commodity.lower()}_prices.csv"
            if not raw_path.exists():
                logger.warning(f"  {commodity}: Raw price file not found — skipping.")
                continue
            df = self._process_single(commodity, raw_path)
            processed[commodity] = df

            out_path = self.proc_dir / f"{commodity.lower()}_prices_processed.csv"
            df.to_csv(out_path)
        return processed

    def _process_single(self, commodity: str, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()

        # ── 1. Handle missing values via time-based interpolation ─────────────
        original_nulls = df["price"].isna().sum()
        df["price"] = df["price"].interpolate(method="time")
        df["price"] = df["price"].fillna(method="bfill").fillna(method="ffill")
        if original_nulls:
            logger.info(f"  {commodity}: Interpolated {original_nulls} missing price values")

        # ── 2. Remove zero / negative prices ─────────────────────────────────
        df = df[df["price"] > 0]

        # ── 3. Compute returns ────────────────────────────────────────────────
        df["daily_return"]   = df["price"].pct_change()
        df["log_return"]     = np.log(df["price"]).diff()

        # Weekly & monthly via resample
        weekly  = df["price"].resample("W").last()
        monthly = df["price"].resample("ME").last()

        df["weekly_return"]  = df["price"].resample("D").last().pct_change(5)
        df["monthly_return"] = df["price"].resample("D").last().pct_change(21)

        # ── 4. Rolling volatility (21-day) ────────────────────────────────────
        df["rolling_vol_21d"] = df["daily_return"].rolling(21).std() * np.sqrt(252)

        # ── 5. Spread (for pairs — computed later in merge step) ──────────────
        df = df.dropna(subset=["daily_return"])
        logger.info(f"  {commodity}: {len(df)} rows after preprocessing")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class WeatherPreprocessor:
    """Loads composite weather CSVs and aligns them to the price date index."""

    def __init__(self, raw_dir: Path, proc_dir: Path):
        self.raw_dir  = raw_dir / "weather"
        self.proc_dir = proc_dir

    def process_all(self) -> dict[str, pd.DataFrame]:
        processed = {}
        for commodity in tqdm(COMMODITIES, desc="Processing weather"):
            weather_path = self.raw_dir / f"{commodity.lower()}_weather.csv"
            if not weather_path.exists():
                logger.warning(f"  {commodity}: Weather file not found — skipping.")
                continue
            df = self._process_single(commodity, weather_path)
            processed[commodity] = df

            out_path = self.proc_dir / f"{commodity.lower()}_weather_processed.csv"
            df.to_csv(out_path)
        return processed

    def _process_single(self, commodity: str, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()

        # Ensure daily frequency
        df = df.asfreq("D")
        df = df.interpolate(method="time")
        df = df.fillna(method="bfill").fillna(method="ffill")

        # Convert precipitation from mm/10 (NOAA encoding) to mm
        if "precipitation" in df.columns:
            df["precipitation"] = df["precipitation"] / 10.0

        # Seasonal decomposition flags
        df["month"]  = df.index.month
        df["season"] = df["month"].map({
            12: "winter", 1: "winter", 2: "winter",
            3: "spring",  4: "spring", 5: "spring",
            6: "summer",  7: "summer", 8: "summer",
            9: "autumn",  10: "autumn",11: "autumn",
        })

        return df


# ══════════════════════════════════════════════════════════════════════════════
# PAIR MERGING
# ══════════════════════════════════════════════════════════════════════════════

class PairMerger:
    """Merges commodity pair prices + weather into unified analysis DataFrames."""

    def __init__(self, proc_dir: Path):
        self.proc_dir = proc_dir

    def merge_all_pairs(
        self,
        prices: dict[str, pd.DataFrame],
        weather: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        pairs = CONFIG["commodity_pairs"]
        merged = {}

        for pair_conf in tqdm(pairs, desc="Merging pairs"):
            base  = pair_conf["base"]
            quote = pair_conf["quote"]
            name  = pair_conf["name"]

            if base not in prices or quote not in prices:
                logger.warning(f"  {name}: Missing price data — skipping.")
                continue

            df = self._merge_pair(base, quote, prices, weather)
            merged[name] = df

            out_name = name.replace("/", "_").lower()
            out_path = self.proc_dir / f"{out_name}_merged.csv"
            df.to_csv(out_path)
            logger.info(f"  {name}: Merged dataset saved → {out_path.name} ({len(df)} rows)")

        return merged

    def _merge_pair(
        self,
        base: str,
        quote: str,
        prices: dict,
        weather: dict,
    ) -> pd.DataFrame:
        p_base  = prices[base][["price", "daily_return", "log_return"]].rename(
            columns=lambda c: f"{base.lower()}_{c}"
        )
        p_quote = prices[quote][["price", "daily_return", "log_return"]].rename(
            columns=lambda c: f"{quote.lower()}_{c}"
        )

        df = p_base.join(p_quote, how="inner")

        # Spread and ratio
        b_price = f"{base.lower()}_price"
        q_price = f"{quote.lower()}_price"
        df["spread"]       = df[b_price] - df[q_price]
        df["spread_pct"]   = df["spread"].pct_change()
        df["ratio"]        = df[b_price] / df[q_price]

        # Attach weather for base commodity
        if base in weather:
            w = weather[base][["temperature", "precipitation"]].rename(
                columns={"temperature": "base_temp", "precipitation": "base_precip"}
            )
            df = df.join(w, how="left")

        df = df.interpolate(method="time")
        df = df.dropna(subset=[b_price, q_price])
        return df


# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_descriptive_stats(
    prices: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Compute and save descriptive statistics table (mirrors original paper Table 4.1)."""
    rows = []
    for commodity, df in prices.items():
        col = "price"
        ret = "daily_return"
        row = {
            "commodity":  commodity,
            "n_obs":      len(df),
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date":   df.index.max().strftime("%Y-%m-%d"),
            "mean_price": round(df[col].mean(), 4),
            "std_price":  round(df[col].std(), 4),
            "min_price":  round(df[col].min(), 4),
            "q25_price":  round(df[col].quantile(0.25), 4),
            "median_price":round(df[col].median(), 4),
            "q75_price":  round(df[col].quantile(0.75), 4),
            "max_price":  round(df[col].max(), 4),
            "mean_daily_return": round(df[ret].mean(), 6),
            "std_daily_return":  round(df[ret].std(), 6),
            "skewness":   round(df[ret].skew(), 4),
            "kurtosis":   round(df[ret].kurtosis(), 4),
        }
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    out_path = output_dir / "descriptive_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    logger.info(f"Descriptive statistics saved → {out_path.name}")
    return stats_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("Starting preprocessing pipeline...")

    # Load existing processed price/weather files or generate placeholders
    price_proc   = PricePreprocessor(RAW_DIR, PROC_DIR)
    weather_proc = WeatherPreprocessor(RAW_DIR, PROC_DIR)

    prices  = price_proc.process_all()
    weather = weather_proc.process_all()

    merger  = PairMerger(PROC_DIR)
    pairs   = merger.merge_all_pairs(prices, weather)

    results_dir = ROOT / CONFIG["paths"]["results_tables"]
    results_dir.mkdir(parents=True, exist_ok=True)
    if prices:
        compute_descriptive_stats(prices, results_dir)

    logger.info(f"✅ Preprocessing complete. {len(pairs)} pairs ready for analysis.")


if __name__ == "__main__":
    main()
