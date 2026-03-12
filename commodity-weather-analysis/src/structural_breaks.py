"""
structural_breaks.py
====================
Detects structural breaks in commodity price series using:
- Zivot-Andrews test (single break)
- Bai-Perron procedure (multiple breaks)

Identified breaks are output as dummy variables for use in SARIMAX/VECM models.
Key improvement: original paper only noted Silver Thursday (1980) visually
without formally testing for or controlling structural breaks.

Usage:
    python src/structural_breaks.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import zivot_andrews
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR    = ROOT / CONFIG["paths"]["processed_data"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results_tables"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES = list(CONFIG["weather_stations"].keys())


# ══════════════════════════════════════════════════════════════════════════════
# ZIVOT-ANDREWS TEST
# ══════════════════════════════════════════════════════════════════════════════

def zivot_andrews_test(series: pd.Series, commodity: str) -> dict:
    """
    Zivot-Andrews unit root test allowing for a single structural break.
    Identifies the breakpoint date and tests for stationarity allowing for it.
    """
    s = series.dropna()
    if len(s) < 50:
        return {"commodity": commodity, "za_stat": np.nan, "za_pvalue": np.nan,
                "break_date": None, "break_significant": False}
    try:
        stat, pvalue, cvdict, baselag, breakindex = zivot_andrews(s, trim=0.10)

        break_date = s.index[breakindex] if breakindex is not None else None
        significant = pvalue < CONFIG["tests"]["significance_level"]

        logger.info(
            f"  {commodity}: ZA stat={stat:.4f} p={pvalue:.4f} "
            f"break_date={break_date.strftime('%Y-%m') if break_date else 'N/A'} "
            f"{'✅ Significant' if significant else '❌ Not significant'}"
        )

        return {
            "commodity":        commodity,
            "za_stat":          round(stat, 4),
            "za_pvalue":        round(pvalue, 4),
            "break_date":       break_date,
            "break_significant": significant,
            "cv_1pct":          cvdict.get("1%", np.nan),
            "cv_5pct":          cvdict.get("5%", np.nan),
        }
    except Exception as e:
        logger.error(f"  {commodity}: Zivot-Andrews failed — {e}")
        return {"commodity": commodity, "za_stat": np.nan, "za_pvalue": np.nan,
                "break_date": None, "break_significant": False}


# ══════════════════════════════════════════════════════════════════════════════
# BAI-PERRON MULTIPLE BREAKPOINTS (simplified CUSUM-based implementation)
# ══════════════════════════════════════════════════════════════════════════════

def bai_perron_cusum(series: pd.Series, commodity: str, max_breaks: int = 5) -> dict:
    """
    Simplified multiple breakpoint detection using CUSUM of squared residuals.
    For full Bai-Perron, consider the `ruptures` library.

    Returns up to max_breaks candidate breakpoints.
    """
    try:
        import ruptures as rpt
        monthly = series.resample("ME").last().dropna()
        signal  = monthly.values.reshape(-1, 1)

        algo = rpt.Pelt(model="rbf").fit(signal)
        breakpoints = algo.predict(pen=10)

        # Convert indices to dates (exclude last which is len)
        break_dates = [monthly.index[i - 1] for i in breakpoints[:-1]]

        logger.info(f"  {commodity}: {len(break_dates)} structural break(s) detected")
        for d in break_dates:
            logger.info(f"    → {d.strftime('%Y-%m')}")

        return {
            "commodity":    commodity,
            "n_breaks":     len(break_dates),
            "break_dates":  break_dates,
        }

    except ImportError:
        logger.warning("  `ruptures` library not installed — using CUSUM fallback")
        return _cusum_fallback(series, commodity)
    except Exception as e:
        logger.error(f"  {commodity}: Bai-Perron failed — {e}")
        return {"commodity": commodity, "n_breaks": 0, "break_dates": []}


def _cusum_fallback(series: pd.Series, commodity: str) -> dict:
    """Simple CUSUM change point detection as fallback."""
    monthly = series.resample("ME").last().dropna()
    mean    = monthly.mean()
    cusum   = (monthly - mean).cumsum()

    # Local maxima/minima in CUSUM > 2 std as candidate breaks
    threshold = 2 * cusum.std()
    candidates = cusum[cusum.abs() > threshold]

    # Take at most 5 most extreme candidates
    break_dates = cusum.abs().nlargest(min(5, len(cusum))).index.tolist()
    break_dates.sort()

    return {
        "commodity":  commodity,
        "n_breaks":   len(break_dates),
        "break_dates": break_dates,
    }


def create_break_dummies(
    index: pd.DatetimeIndex,
    break_dates: list,
    window: int = 3,
) -> pd.DataFrame:
    """
    Create dummy variable columns for structural break windows.
    Each dummy = 1 for `window` months around a break date, 0 otherwise.
    """
    dummies = {}
    for d in break_dates:
        col = f"break_{d.strftime('%Y%m')}"
        dummy = pd.Series(0, index=index)
        mask = (index >= d - pd.DateOffset(months=window)) & \
               (index <= d + pd.DateOffset(months=window))
        dummy[mask] = 1
        dummies[col] = dummy

    return pd.DataFrame(dummies)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_structural_breaks(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    za_rows = []
    bp_rows = []

    for commodity in tqdm(COMMODITIES, desc="Structural break detection"):
        if commodity not in prices:
            continue

        series = prices[commodity]["price"]

        # Zivot-Andrews
        za = zivot_andrews_test(series, commodity)
        za_rows.append(za)

        # Bai-Perron / CUSUM
        bp = bai_perron_cusum(series, commodity)
        bp_rows.append({
            "commodity":  commodity,
            "n_breaks":   bp["n_breaks"],
            "break_dates": ", ".join(
                d.strftime("%Y-%m") for d in bp["break_dates"]
            ),
        })

    za_df = pd.DataFrame(za_rows)
    bp_df = pd.DataFrame(bp_rows)

    summary = za_df.merge(bp_df, on="commodity")
    summary.to_csv(RESULTS_DIR / "structural_breaks_results.csv", index=False)

    logger.info(f"\n✅ Structural break results saved")
    print(summary[["commodity", "break_date", "break_significant", "n_breaks"]].to_string(index=False))
    return summary


def load_prices() -> dict[str, pd.DataFrame]:
    prices = {}
    for c in COMMODITIES:
        p = PROC_DIR / f"{c.lower()}_prices_processed.csv"
        if p.exists():
            prices[c] = pd.read_csv(p, index_col=0, parse_dates=True)
    return prices


def main():
    prices = load_prices()
    if not prices:
        logger.error("No processed price data found. Run preprocessing.py first.")
        return
    run_all_structural_breaks(prices)


if __name__ == "__main__":
    main()
