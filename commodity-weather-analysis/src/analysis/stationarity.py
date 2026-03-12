"""
stationarity.py
---------------
Stationarity testing and structural break detection.

Tests implemented:
    - Augmented Dickey-Fuller (ADF)   — original paper used this
    - KPSS                            — NEW: complementary to ADF
    - Zivot-Andrews                   — NEW: structural break-aware unit root test
    - Bai-Perron multiple breakpoints — NEW: identify all major structural breaks

Improvements over original paper:
    The original only used ADF. Including KPSS avoids the ADF's tendency
    to over-reject stationarity near unit root boundaries. Zivot-Andrews
    and Bai-Perron address structural breaks that were observed visually
    (e.g., Silver Thursday 1980, financial crisis 2008) but not modelled.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import zivot_andrews
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ADF Test (replicates original paper)
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, significance: float = 0.05) -> dict:
    """Augmented Dickey-Fuller unit root test.

    Null hypothesis: Series has a unit root (non-stationary).
    Reject H₀ → series is stationary.

    Parameters
    ----------
    series       : pd.Series of prices or returns
    significance : Significance level (default 0.05)

    Returns
    -------
    dict with test statistic, p-value, critical values, and stationarity conclusion
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, pval, lags, nobs, crit, _ = result

    conclusion = "Stationary" if pval < significance else "Non-Stationary"

    return {
        "test":             "ADF",
        "series":           series.name,
        "test_statistic":   round(stat, 4),
        "p_value":          round(pval, 4),
        "lags_used":        lags,
        "n_obs":            nobs,
        "critical_1pct":    round(crit["1%"], 3),
        "critical_5pct":    round(crit["5%"], 3),
        "critical_10pct":   round(crit["10%"], 3),
        "conclusion":       conclusion,
        "reject_null":      pval < significance,
    }


# ---------------------------------------------------------------------------
# KPSS Test — complementary to ADF
# ---------------------------------------------------------------------------

def kpss_test(series: pd.Series, regression: str = "c", significance: float = 0.05) -> dict:
    """KPSS stationarity test.

    Null hypothesis: Series IS stationary.
    Reject H₀ → series is NOT stationary (opposite to ADF).

    Using ADF + KPSS together:
        ADF reject + KPSS fail-to-reject → Stationary (strong evidence)
        ADF fail   + KPSS reject         → Non-stationary (strong evidence)
        Both reject / Both fail-to-reject → Ambiguous (possibly ARFIMA / fractional)

    Parameters
    ----------
    regression : 'c' (constant) or 'ct' (constant + trend)
    """
    stat, pval, lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
    conclusion = "Non-Stationary" if pval < significance else "Stationary"

    return {
        "test":           "KPSS",
        "series":         series.name,
        "test_statistic": round(stat, 4),
        "p_value":        round(pval, 4),
        "lags_used":      lags,
        "critical_1pct":  round(crit["1%"], 3),
        "critical_5pct":  round(crit["5%"], 3),
        "critical_10pct": round(crit["10%"], 3),
        "conclusion":     conclusion,
        "reject_null":    pval < significance,
    }


# ---------------------------------------------------------------------------
# Zivot-Andrews Test — unit root with structural break
# ---------------------------------------------------------------------------

def zivot_andrews_test(series: pd.Series, significance: float = 0.05) -> dict:
    """Zivot-Andrews test for unit root in presence of a single structural break.

    Identifies the most likely break date endogenously.
    Null hypothesis: Series has a unit root with a structural break.

    Parameters
    ----------
    series       : pd.Series (price level, not returns)
    significance : Significance level
    """
    stat, pval, crit, bp_idx, base = zivot_andrews(series.dropna(), trim=0.15)

    # Map break index to a date if series has DatetimeIndex
    if isinstance(series.index, pd.DatetimeIndex):
        break_date = series.dropna().index[bp_idx]
    else:
        break_date = bp_idx

    conclusion = "Stationary (with break)" if pval < significance else "Non-Stationary"

    return {
        "test":           "Zivot-Andrews",
        "series":         series.name,
        "test_statistic": round(stat, 4),
        "p_value":        round(pval, 4),
        "break_date":     str(break_date)[:10],
        "critical_1pct":  round(crit["1%"], 3),
        "critical_5pct":  round(crit["5%"], 3),
        "critical_10pct": round(crit["10%"], 3),
        "conclusion":     conclusion,
        "reject_null":    pval < significance,
    }


# ---------------------------------------------------------------------------
# Bai-Perron Multiple Structural Break Detection
# ---------------------------------------------------------------------------

def bai_perron_breaks(
    series: pd.Series,
    max_breaks: int = 5,
    min_size: float = 0.15,
) -> dict:
    """Bai-Perron sequential test for multiple structural breaks.

    Fits OLS with mean-shift dummies for each candidate break date and
    selects the number of breaks by BIC minimisation.

    Parameters
    ----------
    series     : pd.Series (price or return series)
    max_breaks : Maximum number of breaks to test
    min_size   : Minimum segment size as fraction of sample (default 15%)

    Returns
    -------
    dict with identified break dates, segment means, and BIC values
    """
    y = series.dropna().values
    n = len(y)
    min_seg = max(int(n * min_size), 10)

    best_bic    = np.inf
    best_breaks = []
    bic_by_k    = {}

    for k in range(0, max_breaks + 1):
        if k == 0:
            resid = y - y.mean()
            sse   = np.sum(resid ** 2)
            k_params = 1
        else:
            # Exhaustive search for k break points — simplified DP approach
            breaks, sse = _find_breaks_dp(y, k, min_seg)
            k_params = k + 1 + k  # means + break dummies

        bic = n * np.log(sse / n) + k_params * np.log(n)
        bic_by_k[k] = round(bic, 2)

        if bic < best_bic:
            best_bic    = bic
            best_breaks = breaks if k > 0 else []
            best_k      = k

    # Map break indices to dates
    clean = series.dropna()
    break_dates = [str(clean.index[b])[:10] for b in best_breaks]

    # Compute segment statistics
    segments   = _segment_stats(clean, best_breaks)

    return {
        "test":           "Bai-Perron",
        "series":         series.name,
        "n_breaks":       best_k,
        "break_dates":    break_dates,
        "break_indices":  best_breaks,
        "bic_by_k":       bic_by_k,
        "best_bic":       round(best_bic, 2),
        "segments":       segments,
    }


def _find_breaks_dp(y: np.ndarray, k: int, min_seg: int):
    """Simple dynamic-programming break finder minimising within-segment SSE."""
    n = len(y)

    @np.vectorize
    def seg_sse(start, end):
        seg = y[start:end]
        return np.sum((seg - seg.mean()) ** 2) if len(seg) >= min_seg else np.inf

    # Build SSE table
    sse_table = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + min_seg, n + 1):
            sse_table[i, j - 1] = seg_sse(i, j)

    # DP
    dp  = np.full((k + 1, n), np.inf)
    ptr = np.zeros((k + 1, n), dtype=int)
    dp[0] = np.array([sse_table[0, j] for j in range(n)])

    for bk in range(1, k + 1):
        for end in range(min_seg * (bk + 1) - 1, n):
            best_val = np.inf
            best_split = end
            for split in range(min_seg * bk - 1, end - min_seg + 1):
                val = dp[bk - 1, split] + sse_table[split + 1, end]
                if val < best_val:
                    best_val   = val
                    best_split = split
            dp[bk, end]  = best_val
            ptr[bk, end] = best_split

    # Backtrack
    breaks = []
    pos = n - 1
    for bk in range(k, 0, -1):
        pos = ptr[bk, pos]
        breaks.insert(0, pos)

    return breaks, dp[k, n - 1]


def _segment_stats(series: pd.Series, break_indices: list) -> list:
    """Compute mean and std for each segment between break points."""
    all_idx = [0] + break_indices + [len(series)]
    segments = []
    for i in range(len(all_idx) - 1):
        seg = series.iloc[all_idx[i]:all_idx[i + 1]]
        segments.append({
            "start": str(seg.index[0])[:10],
            "end":   str(seg.index[-1])[:10],
            "mean":  round(seg.mean(), 4),
            "std":   round(seg.std(), 4),
            "n":     len(seg),
        })
    return segments


# ---------------------------------------------------------------------------
# Comprehensive stationarity report
# ---------------------------------------------------------------------------

def full_stationarity_report(
    df: pd.DataFrame,
    series_type: str = "prices",
) -> pd.DataFrame:
    """Run ADF, KPSS, and Zivot-Andrews on all columns of a DataFrame.

    Parameters
    ----------
    df          : DataFrame (columns = commodities)
    series_type : Label for the report header ('prices' or 'returns')

    Returns
    -------
    pd.DataFrame — one row per commodity with test results
    """
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        adf  = adf_test(series)
        kpss_r = kpss_test(series)
        za   = zivot_andrews_test(series)

        # Combined verdict
        adf_stat = adf["conclusion"]
        kpss_stat = kpss_r["conclusion"]

        if adf_stat == "Stationary" and kpss_stat == "Stationary":
            verdict = "✅ Stationary"
        elif adf_stat == "Non-Stationary" and kpss_stat == "Non-Stationary":
            verdict = "❌ Non-Stationary"
        else:
            verdict = "⚠️ Ambiguous"

        rows.append({
            "Commodity":       col,
            "ADF_stat":        adf["test_statistic"],
            "ADF_pval":        adf["p_value"],
            "ADF_conclusion":  adf["conclusion"],
            "KPSS_stat":       kpss_r["test_statistic"],
            "KPSS_pval":       kpss_r["p_value"],
            "KPSS_conclusion": kpss_r["conclusion"],
            "ZA_break_date":   za["break_date"],
            "ZA_conclusion":   za["conclusion"],
            "Combined":        verdict,
        })

    report = pd.DataFrame(rows).set_index("Commodity")
    print(f"\n{'=' * 60}")
    print(f"  Stationarity Report — {series_type.upper()}")
    print(f"{'=' * 60}")
    print(report.to_string())
    return report


if __name__ == "__main__":
    # Quick demo with synthetic AR(1) data
    np.random.seed(42)
    n = 500
    stationary_series     = pd.Series(np.random.randn(n), name="White_Noise")
    nonstationary_series  = pd.Series(np.cumsum(np.random.randn(n)), name="Random_Walk")

    for s in [stationary_series, nonstationary_series]:
        print(f"\n--- {s.name} ---")
        print("ADF:", adf_test(s)["conclusion"])
        print("KPSS:", kpss_test(s)["conclusion"])
