"""
granger_causality.py
====================
Granger causality tests: does weather (temperature/precipitation)
help predict commodity prices beyond the prices' own history?

Key improvement over original paper: replaces visual overlays and
coefficient p-values with formal directional causality tests.

Usage:
    python src/granger_causality.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR    = ROOT / CONFIG["paths"]["processed_data"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results_tables"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIG      = CONFIG["tests"]["significance_level"]
MAX_LAGS = CONFIG["tests"]["granger_max_lags"]
COMMODITIES = list(CONFIG["weather_stations"].keys())

WEATHER_VARS = ["temperature", "precipitation"]
PRICE_VAR    = "price"


def run_granger_test(
    price: pd.Series,
    weather: pd.Series,
    commodity: str,
    weather_var: str,
    max_lags: int,
) -> dict:
    """
    Test: does `weather_var` Granger-cause `price`?
    Uses the F-test from grangercausalitytests across all lags up to max_lags.
    Reports the best (minimum p-value) lag.
    """
    df = pd.concat([price, weather], axis=1).dropna()
    df.columns = ["price", "weather"]

    # Resample to monthly for stability
    df = df.resample("ME").mean().dropna()

    if len(df) < max_lags * 3:
        return {
            "commodity":   commodity,
            "weather_var": weather_var,
            "best_lag":    np.nan,
            "min_pvalue":  np.nan,
            "f_stat":      np.nan,
            "granger_causes": False,
            "note":        "Insufficient data",
        }

    try:
        # grangercausalitytests returns dict: lag -> {ssr_ftest, ssr_chi2test, ...}
        gc_results = grangercausalitytests(
            df[["price", "weather"]],
            maxlag=max_lags,
            verbose=False,
        )

        # Find lag with minimum F-test p-value
        pvalues = {
            lag: res[0]["ssr_ftest"][1]   # [1] is the p-value
            for lag, res in gc_results.items()
        }
        best_lag = min(pvalues, key=pvalues.get)
        min_pval = pvalues[best_lag]
        f_stat   = gc_results[best_lag][0]["ssr_ftest"][0]

        causes = min_pval < SIG
        sig_label = "✅ Significant" if causes else "❌ Not significant"

        logger.info(
            f"  {commodity} | {weather_var}: best_lag={best_lag} "
            f"F={f_stat:.3f} p={min_pval:.4f} → {sig_label}"
        )

        return {
            "commodity":       commodity,
            "weather_var":     weather_var,
            "best_lag":        best_lag,
            "min_pvalue":      round(min_pval, 4),
            "f_stat":          round(f_stat, 4),
            "granger_causes":  causes,
            "significance":    "5%" if min_pval < 0.05 else ("10%" if min_pval < 0.10 else "ns"),
        }

    except Exception as e:
        logger.error(f"  {commodity} | {weather_var}: Granger test failed — {e}")
        return {
            "commodity":   commodity,
            "weather_var": weather_var,
            "best_lag":    np.nan,
            "min_pvalue":  np.nan,
            "f_stat":      np.nan,
            "granger_causes": False,
            "note":        str(e),
        }


def run_all_granger(
    prices: dict[str, pd.DataFrame],
    weather: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for commodity in tqdm(COMMODITIES, desc="Granger causality tests"):
        if commodity not in prices or commodity not in weather:
            continue

        price_series   = prices[commodity]["price"]
        weather_df     = weather[commodity]

        for w_var in WEATHER_VARS:
            if w_var not in weather_df.columns:
                continue
            row = run_granger_test(
                price_series,
                weather_df[w_var],
                commodity,
                w_var,
                MAX_LAGS,
            )
            rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "granger_causality_results.csv", index=False)

    # ── Pretty print summary table ────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info("GRANGER CAUSALITY SUMMARY (weather → commodity price)")
    logger.info(f"{'='*65}")
    pivot = result_df.pivot_table(
        index="commodity",
        columns="weather_var",
        values="significance",
        aggfunc="first",
    ).fillna("—")
    print(pivot.to_string())

    logger.info(f"\n✅ Granger causality results saved ({len(result_df)} tests)")
    return result_df


def load_prices() -> dict[str, pd.DataFrame]:
    prices = {}
    for c in COMMODITIES:
        p = PROC_DIR / f"{c.lower()}_prices_processed.csv"
        if p.exists():
            prices[c] = pd.read_csv(p, index_col=0, parse_dates=True)
    return prices


def load_weather() -> dict[str, pd.DataFrame]:
    weather = {}
    for c in COMMODITIES:
        p = PROC_DIR / f"{c.lower()}_weather_processed.csv"
        if p.exists():
            weather[c] = pd.read_csv(p, index_col=0, parse_dates=True)
    return weather


def main():
    prices  = load_prices()
    weather = load_weather()
    if not prices or not weather:
        logger.error("Processed data not found. Run preprocessing.py first.")
        return
    run_all_granger(prices, weather)


if __name__ == "__main__":
    main()
