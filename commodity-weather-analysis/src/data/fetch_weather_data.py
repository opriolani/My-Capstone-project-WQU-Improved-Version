"""
fetch_weather_data.py
---------------------
Downloads and processes weather data (temperature & precipitation) from
the NOAA Climate Data Online (CDO) API.

Improvement over original paper:
    - Multi-station, production-weighted regional climate composites
      instead of a single weather station per commodity
    - Supports programmatic download via NOAA CDO REST API
    - Includes regional weighting based on production share
    - Handles missing data with time-based interpolation

NOAA API token:
    Register at https://www.ncdc.noaa.gov/cdo-web/token
    Set as environment variable: NOAA_API_TOKEN=your_token_here
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
RAW_DATA_DIR  = Path(__file__).parents[2] / "data" / "raw"
PROC_DATA_DIR = Path(__file__).parents[2] / "data" / "processed"

# ---------------------------------------------------------------------------
# Station configuration — multiple stations per commodity with production weights
# Improvement: replaces single-station approach from original paper
# ---------------------------------------------------------------------------
COMMODITY_STATIONS = {
    "Corn": [
        # Top 3 producers: USA (35%), China (23%), Brazil (10%)
        {"station": "GHCND:USW00014733", "country": "USA",    "weight": 0.50},
        {"station": "GHCND:CHM00054511", "country": "China",  "weight": 0.30},
        {"station": "GHCND:BRM00083743", "country": "Brazil", "weight": 0.20},
    ],
    "Oats": [
        # Top 3 producers: EU (35%), Russia (19%), Canada (18%)
        {"station": "GHCND:SWE00137987", "country": "Sweden",  "weight": 0.35},
        {"station": "GHCND:RSM00027612", "country": "Russia",  "weight": 0.35},
        {"station": "GHCND:CA005051065", "country": "Canada",  "weight": 0.30},
    ],
    "Wheat": [
        # Top 3 producers: EU, China, India
        {"station": "GHCND:FRM00007150", "country": "France", "weight": 0.40},
        {"station": "GHCND:CHM00054511", "country": "China",  "weight": 0.35},
        {"station": "GHCND:IN022021600", "country": "India",  "weight": 0.25},
    ],
    "Soybean": [
        # Top 3 producers: China, USA, Brazil
        {"station": "GHCND:CHM00054511", "country": "China",  "weight": 0.40},
        {"station": "GHCND:USW00014733", "country": "USA",    "weight": 0.30},
        {"station": "GHCND:BRM00083743", "country": "Brazil", "weight": 0.30},
    ],
    "Coffee": [
        # Top 3 producers: Brazil, Vietnam, Colombia
        {"station": "GHCND:BRM00083743", "country": "Brazil",   "weight": 0.45},
        {"station": "GHCND:VM000048820", "country": "Vietnam",  "weight": 0.30},
        {"station": "GHCND:COM00080222", "country": "Colombia", "weight": 0.25},
    ],
    "Cocoa": [
        # Top 3 producers: Côte d'Ivoire, Ghana, Indonesia
        {"station": "GHCND:IVM00065578", "country": "CoteDIvoire", "weight": 0.40},
        {"station": "GHCND:GH000065418", "country": "Ghana",       "weight": 0.35},
        {"station": "GHCND:IDM00096933", "country": "Indonesia",   "weight": 0.25},
    ],
    "Gold": [
        # Top 3 producers: China, Australia, Russia
        {"station": "GHCND:CHM00054511", "country": "China",     "weight": 0.40},
        {"station": "GHCND:ASN00066037", "country": "Australia", "weight": 0.35},
        {"station": "GHCND:RSM00027612", "country": "Russia",    "weight": 0.25},
    ],
    "Silver": [
        # Top 3 producers: Mexico, Peru, China
        {"station": "GHCND:MXM00076525", "country": "Mexico", "weight": 0.40},
        {"station": "GHCND:PE000084401", "country": "Peru",   "weight": 0.35},
        {"station": "GHCND:CHM00054511", "country": "China",  "weight": 0.25},
    ],
}


# ---------------------------------------------------------------------------
# NOAA CDO API fetcher
# ---------------------------------------------------------------------------

def _fetch_noaa(
    station_id: str,
    start: str,
    end: str,
    datatypes: list = None,
    token: str = None,
) -> pd.DataFrame:
    """Fetch weather data from NOAA CDO REST API for a single station.

    Parameters
    ----------
    station_id : NOAA station identifier (e.g., 'GHCND:USW00014733')
    start      : Start date (YYYY-MM-DD)
    end        : End date   (YYYY-MM-DD)
    datatypes  : List of NOAA data types (default: TMAX, TMIN, PRCP)
    token      : NOAA API token (reads NOAA_API_TOKEN env var if None)

    Returns
    -------
    pd.DataFrame with columns: date, TMAX, TMIN, TAVG, PRCP
    """
    if token is None:
        token = os.getenv("NOAA_API_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "NOAA API token not found. Set NOAA_API_TOKEN environment variable.\n"
            "Register at: https://www.ncdc.noaa.gov/cdo-web/token"
        )

    if datatypes is None:
        datatypes = ["TMAX", "TMIN", "PRCP"]

    headers = {"token": token}
    records = []
    offset  = 1

    # NOAA CDO API paginates at 1000 records per request — handle chunking by year
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)

    for year in range(start_dt.year, end_dt.year + 1):
        y_start = max(start_dt, pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        y_end   = min(end_dt,   pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")

        params = {
            "datasetid":  "GHCND",
            "stationid":  station_id,
            "startdate":  y_start,
            "enddate":    y_end,
            "datatypeid": ",".join(datatypes),
            "units":      "metric",
            "limit":      1000,
            "offset":     offset,
        }

        try:
            resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "results" in data:
                records.extend(data["results"])
            time.sleep(0.2)  # Respect rate limit: 5 requests/sec
        except requests.RequestException as e:
            logger.warning(f"  API error for {station_id} year {year}: {e}")
            continue

    if not records:
        logger.warning(f"  No records returned for station {station_id}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean")
    df.columns.name = None

    # Compute average temperature from TMAX and TMIN if available
    if "TMAX" in df.columns and "TMIN" in df.columns:
        df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2.0
    elif "TAVG" not in df.columns:
        df["TAVG"] = np.nan

    # Rename PRCP for clarity
    if "PRCP" not in df.columns:
        df["PRCP"] = np.nan

    return df[["TAVG", "PRCP"]]


# ---------------------------------------------------------------------------
# Weighted composite weather builder
# ---------------------------------------------------------------------------

def fetch_weighted_weather(
    commodity: str,
    start: str = "1973-01-01",
    end: str = "2024-12-31",
    token: str = None,
) -> pd.DataFrame:
    """Fetch multi-station, production-weighted weather composite for a commodity.

    This is the key improvement over the original paper (single station per commodity).
    Each station's temperature and precipitation are weighted by the country's
    share of global production before averaging.

    Parameters
    ----------
    commodity : Name of commodity (must be key in COMMODITY_STATIONS)
    start     : Start date
    end       : End date
    token     : NOAA API token

    Returns
    -------
    pd.DataFrame with columns: TAVG (°C), PRCP (mm)
    """
    stations = COMMODITY_STATIONS.get(commodity)
    if not stations:
        raise ValueError(f"No station config for commodity: {commodity}")

    logger.info(f"Fetching weather for {commodity} ({len(stations)} stations)")
    weighted_tavg = []
    weighted_prcp = []
    total_weight  = 0.0

    for cfg in stations:
        df = _fetch_noaa(cfg["station"], start, end, token=token)
        if df.empty:
            logger.warning(f"  Skipping {cfg['country']} (no data)")
            continue
        w = cfg["weight"]
        weighted_tavg.append(df["TAVG"] * w)
        weighted_prcp.append(df["PRCP"] * w)
        total_weight += w

    if not weighted_tavg:
        logger.error(f"No weather data retrieved for {commodity}")
        return pd.DataFrame()

    # Normalise weights in case some stations returned no data
    composite_tavg = sum(weighted_tavg) / total_weight
    composite_prcp = sum(weighted_prcp) / total_weight

    composite = pd.DataFrame({
        "TAVG": composite_tavg,
        "PRCP": composite_prcp,
    })
    composite = _interpolate_weather(composite)
    return composite


def fetch_all_weather(
    start: str = "1973-01-01",
    end: str = "2024-12-31",
    save: bool = True,
    token: str = None,
) -> dict:
    """Fetch weighted weather composites for all eight commodities.

    Returns
    -------
    dict mapping commodity name → pd.DataFrame (TAVG, PRCP)
    """
    weather = {}
    for commodity in COMMODITY_STATIONS.keys():
        df = fetch_weighted_weather(commodity, start, end, token=token)
        weather[commodity] = df
        if save and not df.empty:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = RAW_DATA_DIR / f"weather_{commodity.lower()}.csv"
            df.to_csv(path)
            logger.info(f"  Saved → {path}")
    return weather


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _interpolate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing weather observations using time-based method."""
    df = df.copy()
    df.interpolate(method="time", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def align_weather_to_prices(
    prices: pd.DataFrame,
    weather: dict,
) -> dict:
    """Reindex weather DataFrames to match trading days in prices DataFrame.

    Parameters
    ----------
    prices  : Commodity prices DataFrame
    weather : Dict of commodity → weather DataFrame

    Returns
    -------
    Dict of commodity → weather DataFrame aligned to prices.index
    """
    aligned = {}
    for commodity, wdf in weather.items():
        if wdf.empty:
            aligned[commodity] = wdf
            continue
        wdf = wdf.reindex(prices.index, method="ffill")
        aligned[commodity] = wdf
    return aligned


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_weather(commodity: str) -> pd.DataFrame:
    """Load saved weather CSV for a commodity."""
    path = RAW_DATA_DIR / f"weather_{commodity.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {path}\n"
            f"Run fetch_all_weather() first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_all_weather() -> dict:
    """Load all saved weather files into a dict."""
    return {c: load_weather(c) for c in COMMODITY_STATIONS.keys()}


if __name__ == "__main__":
    # Demo: show station configuration summary
    print("\nStation Configuration Summary")
    print("=" * 60)
    for commodity, stations in COMMODITY_STATIONS.items():
        countries = [f"{s['country']} ({s['weight']:.0%})" for s in stations]
        print(f"  {commodity:<10}: {', '.join(countries)}")
