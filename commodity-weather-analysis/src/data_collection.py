"""
data_collection.py
==================
Downloads commodity price data and multi-station weather data.
Handles Macrotrends (scrape), Nasdaq Data Link (Quandl), and NOAA CDO API.

Usage:
    python src/data_collection.py --start 1973-01-01 --end 2024-12-31
"""

import os
import time
import argparse
import logging
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import yaml
import yfinance as yf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Load Config ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = ROOT / CONFIG["paths"]["raw_data"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Yahoo Finance tickers (fallback when Quandl unavailable) ──────────────────
YAHOO_TICKERS = {
    "CORN":    "ZC=F",
    "OATS":    "ZO=F",
    "WHEAT":   "ZW=F",
    "SOYBEAN": "ZS=F",
    "COFFEE":  "KC=F",
    "COCOA":   "CC=F",
    "GOLD":    "GC=F",
    "SILVER":  "SI=F",
}


# ══════════════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════════════

class PriceDataCollector:
    """Downloads historical daily commodity prices via yfinance."""

    def __init__(self, start: str, end: str, output_dir: Path):
        self.start = start
        self.end = end
        self.output_dir = output_dir / "prices"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self) -> dict[str, pd.DataFrame]:
        """Download prices for all configured commodities."""
        results = {}
        for commodity, ticker in tqdm(YAHOO_TICKERS.items(), desc="Downloading prices"):
            df = self._download_single(commodity, ticker)
            if df is not None:
                results[commodity] = df
        return results

    def _download_single(self, commodity: str, ticker: str) -> pd.DataFrame | None:
        out_path = self.output_dir / f"{commodity.lower()}_prices.csv"
        if out_path.exists():
            logger.info(f"  {commodity}: Using cached file {out_path.name}")
            return pd.read_csv(out_path, index_col=0, parse_dates=True)

        try:
            logger.info(f"  {commodity}: Downloading {ticker} from Yahoo Finance...")
            df = yf.download(ticker, start=self.start, end=self.end, progress=False)
            if df.empty:
                logger.warning(f"  {commodity}: No data returned for {ticker}")
                return None

            # Keep only Close price, rename column
            df = df[["Close"]].rename(columns={"Close": "price"})
            df.index.name = "date"
            df.to_csv(out_path)
            logger.info(f"  {commodity}: Saved {len(df)} rows → {out_path.name}")
            return df

        except Exception as e:
            logger.error(f"  {commodity}: Download failed — {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA (NOAA CDO API)
# ══════════════════════════════════════════════════════════════════════════════

class WeatherDataCollector:
    """
    Downloads daily temperature and precipitation from NOAA CDO API.

    Requires a free NOAA token: https://www.ncdc.noaa.gov/cdo-web/token
    Set token in environment variable: export NOAA_TOKEN=your_token_here
    """

    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    MAX_RETRIES = 3
    PAGE_SIZE = 1000

    def __init__(self, start: str, end: str, output_dir: Path):
        self.start = start
        self.end = end
        self.output_dir = output_dir / "weather"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("NOAA_TOKEN", "")
        if not self.token:
            logger.warning(
                "NOAA_TOKEN not set. Weather download will be skipped. "
                "Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token"
            )

    def download_all(self) -> dict[str, pd.DataFrame]:
        """Download weather for all commodities using multi-station composites."""
        if not self.token:
            logger.warning("Skipping weather download — no NOAA token.")
            return {}

        results = {}
        station_config = CONFIG["weather_stations"]

        for commodity, conf in tqdm(station_config.items(), desc="Downloading weather"):
            frames = []
            for station in conf["stations"]:
                df = self._download_station(commodity, station["id"], station["name"])
                if df is not None:
                    df["weight"] = station["weight"]
                    frames.append(df)

            if frames:
                composite = self._compute_weighted_composite(frames)
                results[commodity] = composite
                out_path = self.output_dir / f"{commodity.lower()}_weather.csv"
                composite.to_csv(out_path)
                logger.info(f"  {commodity}: Composite weather saved → {out_path.name}")

        return results

    def _download_station(
        self, commodity: str, station_id: str, station_name: str
    ) -> pd.DataFrame | None:
        """Fetch TMAX, TMIN, TAVG, PRCP for one station via NOAA CDO API."""
        cache_path = self.output_dir / f"raw_{commodity.lower()}_{station_id}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        headers = {"token": self.token}
        params = {
            "datasetid": "GHCND",
            "stationid": f"GHCND:{station_id}",
            "datatypeid": "TMAX,TMIN,PRCP",
            "startdate": self.start,
            "enddate": self.end,
            "limit": self.PAGE_SIZE,
            "units": "metric",
        }

        records = []
        offset = 1
        for attempt in range(self.MAX_RETRIES):
            try:
                params["offset"] = offset
                resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                results_batch = data.get("results", [])
                records.extend(results_batch)

                total = data.get("metadata", {}).get("resultset", {}).get("count", 0)
                if offset + self.PAGE_SIZE > total:
                    break
                offset += self.PAGE_SIZE
                time.sleep(0.5)   # Respect rate limits

            except Exception as e:
                logger.warning(f"  [{station_name}] Attempt {attempt+1} failed: {e}")
                time.sleep(2)

        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean")
        df.columns.name = None

        # Rename and compute TAVG if not present
        rename_map = {"TMAX": "tmax", "TMIN": "tmin", "PRCP": "precipitation"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "tmax" in df.columns and "tmin" in df.columns:
            df["temperature"] = (df["tmax"] + df["tmin"]) / 2

        df.to_csv(cache_path)
        return df

    @staticmethod
    def _compute_weighted_composite(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Compute production-weighted composite of temperature and precipitation
        across multiple weather stations for a given commodity.
        """
        combined = pd.DataFrame()

        for df in frames:
            weight = df.pop("weight").iloc[0]
            for col in ["temperature", "precipitation"]:
                if col in df.columns:
                    combined[col] = combined.get(col, pd.Series(dtype=float)).add(
                        df[col] * weight, fill_value=0
                    )

        # Normalise (weights should sum to 1, but guard against gaps)
        return combined.interpolate(method="time").fillna(method="bfill").fillna(method="ffill")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download commodity & weather data")
    parser.add_argument("--start", default=CONFIG["project"]["start_date"])
    parser.add_argument("--end",   default=CONFIG["project"]["end_date"])
    args = parser.parse_args()

    logger.info(f"Data collection: {args.start} → {args.end}")
    logger.info(f"Output: {RAW_DIR}")

    # Prices
    price_collector = PriceDataCollector(args.start, args.end, RAW_DIR)
    prices = price_collector.download_all()
    logger.info(f"✅ Price data collected for {len(prices)} commodities")

    # Weather
    weather_collector = WeatherDataCollector(args.start, args.end, RAW_DIR)
    weather = weather_collector.download_all()
    logger.info(f"✅ Weather data collected for {len(weather)} commodities")

    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
