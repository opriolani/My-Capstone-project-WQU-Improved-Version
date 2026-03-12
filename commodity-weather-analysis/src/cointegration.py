"""
cointegration.py
================
Engle-Granger two-step and Johansen cointegration tests for all commodity pairs.
Improves on the original paper which only used correlation to assess pair relationships.

Usage:
    python src/cointegration.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR    = ROOT / CONFIG["paths"]["processed_data"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results_tables"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIG = CONFIG["tests"]["significance_level"]


# ══════════════════════════════════════════════════════════════════════════════
# ADF STATIONARITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def adf_test(series: pd.Series, name: str = "") -> dict:
    """
    Augmented Dickey-Fuller test for unit roots.
    Returns a dict with test stat, p-value, and stationarity conclusion.
    """
    result = adfuller(series.dropna(), maxlags=CONFIG["tests"]["adf_max_lags"], autolag="AIC")
    stat, pvalue, lags, nobs, crit_vals, _ = result

    conclusion = "Stationary" if pvalue < SIG else "Non-Stationary"
    return {
        "series":      name,
        "adf_stat":    round(stat, 4),
        "p_value":     round(pvalue, 4),
        "lags_used":   lags,
        "n_obs":       nobs,
        "crit_1pct":   round(crit_vals["1%"], 4),
        "crit_5pct":   round(crit_vals["5%"], 4),
        "crit_10pct":  round(crit_vals["10%"], 4),
        "conclusion":  conclusion,
    }


def run_adf_all_pairs(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run ADF on price levels and first differences for all pair series."""
    rows = []
    for pair_name, df in pairs.items():
        base, quote = pair_name.split("/")
        for commodity in [base, quote]:
            col = f"{commodity.lower()}_price"
            if col not in df.columns:
                continue

            # Level
            rows.append({
                "pair": pair_name,
                "series_type": "level",
                **adf_test(df[col], name=commodity),
            })
            # First difference
            rows.append({
                "pair": pair_name,
                "series_type": "first_diff",
                **adf_test(df[col].diff().dropna(), name=f"Δ{commodity}"),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "adf_stationarity_results.csv", index=False)
    logger.info(f"ADF results saved ({len(result_df)} tests)")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# ENGLE-GRANGER COINTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def engle_granger_test(
    y: pd.Series, x: pd.Series, pair_name: str
) -> dict:
    """
    Engle-Granger two-step cointegration test.
    H0: No cointegration (series are NOT cointegrated).
    """
    # Align series
    aligned = pd.concat([y, x], axis=1).dropna()
    if len(aligned) < 50:
        return {"pair": pair_name, "eg_stat": np.nan, "eg_pvalue": np.nan, "conclusion": "Insufficient data"}

    t_stat, p_value, crit_vals = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])

    conclusion = "Cointegrated" if p_value < SIG else "Not Cointegrated"
    return {
        "pair":        pair_name,
        "eg_stat":     round(t_stat, 4),
        "eg_pvalue":   round(p_value, 4),
        "crit_1pct":   round(crit_vals[0], 4),
        "crit_5pct":   round(crit_vals[1], 4),
        "crit_10pct":  round(crit_vals[2], 4),
        "conclusion":  conclusion,
    }


# ══════════════════════════════════════════════════════════════════════════════
# JOHANSEN COINTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def johansen_test(
    y: pd.Series, x: pd.Series, pair_name: str
) -> dict:
    """
    Johansen cointegration test.
    Tests for the number of cointegrating vectors (rank).
    det_order=0: constant in cointegrating equation
    k_ar_diff=1: first-order VECM
    """
    det_order = CONFIG["tests"]["johansen_det_order"]
    k_ar_diff = CONFIG["tests"]["johansen_k_ar_diff"]

    aligned = pd.concat([y, x], axis=1).dropna()
    if len(aligned) < 100:
        return {
            "pair":         pair_name,
            "trace_stat_r0": np.nan,
            "trace_cv_r0_5pct": np.nan,
            "rank":         np.nan,
            "conclusion":   "Insufficient data",
        }

    result = coint_johansen(aligned.values, det_order, k_ar_diff)

    # Trace statistic for H0: rank=0 (no cointegration)
    trace_r0 = result.lr1[0]
    cv_r0_5pct = result.cvt[0, 1]   # 5% critical value

    # Determine rank
    rank = 0
    for i in range(len(result.lr1)):
        if result.lr1[i] > result.cvt[i, 1]:
            rank = i + 1
        else:
            break

    conclusion = f"Cointegrated (rank={rank})" if rank > 0 else "Not Cointegrated"

    return {
        "pair":                pair_name,
        "trace_stat_r0":       round(trace_r0, 4),
        "trace_cv_r0_5pct":    round(cv_r0_5pct, 4),
        "max_eigen_r0":        round(result.lr2[0], 4),
        "max_eigen_cv_r0_5pct": round(result.cvm[0, 1], 4),
        "rank":                rank,
        "conclusion":          conclusion,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED COINTEGRATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_cointegration_analysis(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run both Engle-Granger and Johansen tests for all pairs.
    Returns a consolidated summary DataFrame.
    """
    eg_rows = []
    joh_rows = []

    for pair_name, df in tqdm(pairs.items(), desc="Cointegration tests"):
        base, quote = pair_name.split("/")
        base_col  = f"{base.lower()}_price"
        quote_col = f"{quote.lower()}_price"

        if base_col not in df.columns or quote_col not in df.columns:
            logger.warning(f"  {pair_name}: Price columns not found in merged data")
            continue

        y = df[base_col].dropna()
        x = df[quote_col].dropna()

        eg_row  = engle_granger_test(y, x, pair_name)
        joh_row = johansen_test(y, x, pair_name)

        eg_rows.append(eg_row)
        joh_rows.append(joh_row)

        logger.info(
            f"  {pair_name}: EG → {eg_row['conclusion']} | "
            f"Johansen → {joh_row['conclusion']}"
        )

    eg_df  = pd.DataFrame(eg_rows)
    joh_df = pd.DataFrame(joh_rows)

    # Merge into summary
    summary = eg_df.merge(
        joh_df[["pair", "rank", "trace_stat_r0", "trace_cv_r0_5pct",
                "max_eigen_r0", "max_eigen_cv_r0_5pct", "conclusion"]],
        on="pair",
        suffixes=("_eg", "_johansen"),
    )

    summary.to_csv(RESULTS_DIR / "cointegration_results.csv", index=False)
    logger.info(f"\n{'='*60}")
    logger.info("COINTEGRATION SUMMARY")
    logger.info(f"{'='*60}")
    print(summary[["pair", "conclusion_eg", "conclusion_johansen"]].to_string(index=False))

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_pairs() -> dict[str, pd.DataFrame]:
    """Load all merged pair DataFrames from processed data directory."""
    pairs = {}
    for pair_conf in CONFIG["commodity_pairs"]:
        name      = pair_conf["name"]
        file_name = name.replace("/", "_").lower() + "_merged.csv"
        path      = PROC_DIR / file_name
        if path.exists():
            pairs[name] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            logger.warning(f"Merged file not found: {file_name} — run preprocessing first.")
    return pairs


def main():
    pairs = load_pairs()
    if not pairs:
        logger.error("No processed pair data found. Run preprocessing.py first.")
        return

    # ADF stationarity
    adf_df = run_adf_all_pairs(pairs)

    # Cointegration
    coint_df = run_cointegration_analysis(pairs)

    logger.info("✅ Cointegration analysis complete.")


if __name__ == "__main__":
    main()
