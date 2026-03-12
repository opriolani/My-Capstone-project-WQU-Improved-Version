"""
cointegration.py
----------------
Cointegration testing and Granger causality analysis.

Tests implemented:
    - Engle-Granger two-step cointegration test
    - Johansen cointegration test
    - Granger causality (weather → commodity price)
    - Impulse Response Functions (for VECM pairs)

Improvements over original paper:
    The original study only used correlation matrices to assess pair relationships.
    Cointegration tests formally determine whether pairs share a long-run
    stochastic trend — a much stronger and statistically grounded claim.
    Granger causality replaces visual inspection of weather overlays.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

warnings.filterwarnings("ignore")

COMMODITY_PAIRS = [
    ("Corn",    "Oats"),
    ("Wheat",   "Soybean"),
    ("Coffee",  "Cocoa"),
    ("Gold",    "Silver"),
]


# ---------------------------------------------------------------------------
# Engle-Granger Cointegration
# ---------------------------------------------------------------------------

def engle_granger_test(
    series_a: pd.Series,
    series_b: pd.Series,
    significance: float = 0.05,
) -> dict:
    """Engle-Granger two-step cointegration test.

    Step 1: Estimate OLS regression of A on B.
    Step 2: Test residuals for stationarity with ADF.

    Null hypothesis: No cointegration (residuals have unit root).
    Reject H₀ → series ARE cointegrated.

    Parameters
    ----------
    series_a, series_b : pd.Series of log prices (same length, aligned)
    significance       : Significance level (default 0.05)
    """
    # Align
    df = pd.concat([series_a, series_b], axis=1).dropna()
    a_name, b_name = df.columns

    stat, pval, crit = coint(df[a_name], df[b_name])

    # OLS for hedge ratio (cointegrating coefficient)
    X = sm.add_constant(df[b_name])
    ols = sm.OLS(df[a_name], X).fit()
    hedge_ratio = ols.params[b_name]
    spread = df[a_name] - hedge_ratio * df[b_name]

    conclusion = "Cointegrated" if pval < significance else "Not Cointegrated"

    return {
        "test":            "Engle-Granger",
        "pair":            f"{a_name} / {b_name}",
        "test_statistic":  round(stat, 4),
        "p_value":         round(pval, 4),
        "critical_1pct":   round(crit[0], 3),
        "critical_5pct":   round(crit[1], 3),
        "critical_10pct":  round(crit[2], 3),
        "hedge_ratio":     round(hedge_ratio, 6),
        "spread_mean":     round(spread.mean(), 4),
        "spread_std":      round(spread.std(), 4),
        "conclusion":      conclusion,
        "cointegrated":    pval < significance,
        "spread":          spread,
    }


# ---------------------------------------------------------------------------
# Johansen Cointegration Test
# ---------------------------------------------------------------------------

def johansen_test(
    df_pair: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance_level: str = "5%",
) -> dict:
    """Johansen maximum likelihood cointegration test.

    Tests the number of cointegrating vectors (rank) in a VAR system.
    More powerful than Engle-Granger for systems with more than 2 series.

    Parameters
    ----------
    df_pair          : DataFrame with 2 columns (aligned price series)
    det_order        : Deterministic terms: -1 (none), 0 (constant), 1 (trend)
    k_ar_diff        : Number of lagged differences in VECM
    significance_level: '1%', '5%', or '10%'

    Returns
    -------
    dict with trace and max-eigenvalue test results
    """
    result = coint_johansen(df_pair.dropna(), det_order, k_ar_diff)

    sig_idx = {"10%": 0, "5%": 1, "1%": 2}[significance_level]
    cols = df_pair.columns.tolist()

    trace_stats    = result.lr1
    trace_crits    = result.cvt[:, sig_idx]
    maxeig_stats   = result.lr2
    maxeig_crits   = result.cvm[:, sig_idx]

    # Determine rank (number of cointegrating relations)
    cointegration_rank = 0
    for i in range(len(trace_stats)):
        if trace_stats[i] > trace_crits[i]:
            cointegration_rank += 1

    rows = []
    hypotheses = [f"r = {i}" for i in range(len(trace_stats))]
    for i, h in enumerate(hypotheses):
        rows.append({
            "H₀":                 h,
            "Trace_stat":         round(trace_stats[i], 3),
            "Trace_crit":         round(trace_crits[i], 3),
            "Trace_reject":       trace_stats[i] > trace_crits[i],
            "MaxEig_stat":        round(maxeig_stats[i], 3),
            "MaxEig_crit":        round(maxeig_crits[i], 3),
            "MaxEig_reject":      maxeig_stats[i] > maxeig_crits[i],
        })

    conclusion = (
        f"Cointegrated (rank = {cointegration_rank})"
        if cointegration_rank > 0
        else "Not Cointegrated"
    )

    return {
        "test":                "Johansen",
        "pair":                f"{cols[0]} / {cols[1]}",
        "significance_level":  significance_level,
        "cointegration_rank":  cointegration_rank,
        "conclusion":          conclusion,
        "cointegrated":        cointegration_rank > 0,
        "eigenvalues":         result.eig.tolist(),
        "cointegrating_vector": result.evec[:, 0].tolist(),
        "results_table":       pd.DataFrame(rows),
    }


# ---------------------------------------------------------------------------
# Run both tests for all commodity pairs
# ---------------------------------------------------------------------------

def run_all_cointegration_tests(
    prices: pd.DataFrame,
    use_log: bool = True,
) -> pd.DataFrame:
    """Run Engle-Granger and Johansen tests for all four commodity pairs.

    Parameters
    ----------
    prices  : DataFrame of daily prices
    use_log : If True, test log prices (recommended for cointegration)

    Returns
    -------
    Summary DataFrame with results for all pairs
    """
    if use_log:
        data = np.log(prices.replace(0, np.nan))
        data.columns = prices.columns
    else:
        data = prices.copy()

    rows = []
    for a, b in COMMODITY_PAIRS:
        if a not in data.columns or b not in data.columns:
            continue

        eg  = engle_granger_test(data[a], data[b])
        joh = johansen_test(data[[a, b]])

        rows.append({
            "Pair":          f"{a} / {b}",
            "EG_pvalue":     eg["p_value"],
            "EG_conclusion": eg["conclusion"],
            "JO_rank":       joh["cointegration_rank"],
            "JO_conclusion": joh["conclusion"],
            "Hedge_ratio":   eg["hedge_ratio"],
            "Cointegrated":  eg["cointegrated"] or joh["cointegrated"],
        })

        print(f"\n{'─' * 50}")
        print(f"  {a} / {b}")
        print(f"  Engle-Granger p={eg['p_value']}  → {eg['conclusion']}")
        print(f"  Johansen rank={joh['cointegration_rank']}  → {joh['conclusion']}")
        print(joh["results_table"].to_string(index=False))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Granger Causality: Weather → Price
# ---------------------------------------------------------------------------

def granger_causality_weather(
    price_series: pd.Series,
    weather_series: pd.Series,
    max_lags: int = 12,
    significance: float = 0.05,
) -> dict:
    """Test whether weather Granger-causes commodity prices.

    Null hypothesis: Weather does NOT Granger-cause prices.
    Reject H₀ → past weather values contain useful information
                 for forecasting prices beyond prices' own history.

    Improvement over original paper: replaces visual inspection of overlaid plots.

    Parameters
    ----------
    price_series   : Commodity return or price series
    weather_series : Temperature or precipitation series
    max_lags       : Maximum lag order to test
    significance   : Significance level

    Returns
    -------
    dict with per-lag p-values and overall conclusion
    """
    # Align and combine
    df = pd.concat(
        [price_series.rename("price"), weather_series.rename("weather")],
        axis=1
    ).dropna()

    if len(df) < max_lags * 3:
        return {"error": "Insufficient data for Granger test"}

    results = grangercausalitytests(df[["price", "weather"]], maxlag=max_lags, verbose=False)

    pvals = {}
    for lag, res in results.items():
        # F-test p-value
        pvals[lag] = round(res[0]["ssr_ftest"][1], 4)

    # Find minimum p-value and corresponding lag
    min_lag  = min(pvals, key=pvals.get)
    min_pval = pvals[min_lag]
    conclusion = (
        "Weather Granger-causes Price" if min_pval < significance
        else "No Granger causality"
    )

    return {
        "test":             "Granger Causality",
        "cause":            weather_series.name,
        "effect":           price_series.name,
        "pvalues_by_lag":   pvals,
        "min_pvalue":       min_pval,
        "optimal_lag":      min_lag,
        "significance":     significance,
        "conclusion":       conclusion,
        "causality_found":  min_pval < significance,
    }


def run_all_granger_tests(
    returns: pd.DataFrame,
    weather: dict,
    max_lags: int = 12,
) -> pd.DataFrame:
    """Run Granger causality tests for all commodities × {temperature, precipitation}.

    Parameters
    ----------
    returns : DataFrame of commodity daily returns
    weather : Dict of commodity → DataFrame with columns TAVG, PRCP
    max_lags: Maximum lags for Granger test

    Returns
    -------
    pd.DataFrame — summary of all causality tests
    """
    rows = []
    for commodity in returns.columns:
        if commodity not in weather:
            continue

        w = weather[commodity]
        price_ret = returns[commodity]

        for var, col in [("Temperature", "TAVG"), ("Precipitation", "PRCP")]:
            if col not in w.columns:
                continue

            weather_s = w[col].reindex(price_ret.index).ffill()
            result = granger_causality_weather(
                price_ret, weather_s.rename(f"{commodity}_{var}"), max_lags=max_lags
            )

            if "error" in result:
                continue

            rows.append({
                "Commodity":     commodity,
                "Weather_var":   var,
                "Min_pvalue":    result["min_pvalue"],
                "Optimal_lag":   result["optimal_lag"],
                "Causality":     "✅ Yes" if result["causality_found"] else "❌ No",
                "Conclusion":    result["conclusion"],
            })

    summary = pd.DataFrame(rows)
    print("\nGranger Causality Summary — Weather → Commodity Returns")
    print("=" * 65)
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Correlation matrix (replicates original paper Figure 4.2)
# ---------------------------------------------------------------------------

def correlation_report(
    returns: pd.DataFrame,
    freq_label: str = "Daily",
) -> pd.DataFrame:
    """Compute and print correlation matrix for commodity returns."""
    corr = returns.corr().round(3)
    print(f"\nCorrelation Matrix — {freq_label} Returns")
    print("=" * 60)
    print(corr.to_string())
    return corr


if __name__ == "__main__":
    # Demo with random correlated data
    np.random.seed(42)
    n = 2000
    gold   = pd.Series(np.cumsum(np.random.randn(n) * 0.01), name="Gold")
    silver = gold * 0.8 + pd.Series(np.random.randn(n) * 0.005)
    silver.name = "Silver"

    eg = engle_granger_test(gold, silver)
    print(f"Gold/Silver Engle-Granger: {eg['conclusion']} (p={eg['p_value']})")

    prices = pd.DataFrame({"Gold": gold, "Silver": silver})
    joh = johansen_test(prices)
    print(f"Gold/Silver Johansen: {joh['conclusion']} (rank={joh['cointegration_rank']})")
