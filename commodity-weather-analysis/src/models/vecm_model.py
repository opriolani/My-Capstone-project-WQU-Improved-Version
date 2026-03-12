"""
vecm_model.py
=============
Vector Error Correction Model (VECM) for cointegrated commodity pairs.
Key improvement: captures long-run equilibrium and adjustment dynamics
that SARIMAX cannot model on individual price series.

Usage:
    python src/models/vecm_model.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from tqdm import tqdm

from utils.metrics import mape, rmse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR    = ROOT / CONFIG["paths"]["processed_data"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results_tables"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pairs confirmed cointegrated (from cointegration.py results)
COINTEGRATED_PAIRS = {"Gold/Silver", "Corn/Oats"}


class VECMModel:
    """
    Fits a VECM for a cointegrated commodity pair.

    Provides:
    - Cointegrating vector (long-run equilibrium)
    - Adjustment coefficients (speed of mean reversion)
    - Short-run dynamics
    - Out-of-sample forecast evaluation
    """

    def __init__(self, pair_name: str, df: pd.DataFrame):
        self.pair_name = pair_name
        self.df        = df.copy()
        self.base, self.quote = pair_name.split("/")
        self.results   = {}

    def fit(self) -> dict:
        base_col  = f"{self.base.lower()}_price"
        quote_col = f"{self.quote.lower()}_price"

        if base_col not in self.df.columns or quote_col not in self.df.columns:
            logger.warning(f"  {self.pair_name}: Required price columns missing")
            return {}

        data = self.df[[base_col, quote_col]].dropna()

        # ── Resample to monthly for VECM stability ───────────────────────────
        data_monthly = data.resample("ME").last().dropna()

        if len(data_monthly) < 60:
            logger.warning(f"  {self.pair_name}: Insufficient data for VECM ({len(data_monthly)} months)")
            return {}

        # ── Train/test split ─────────────────────────────────────────────────
        test_n = CONFIG["ml"]["forecast_horizon"]
        train  = data_monthly.iloc[:-test_n]
        test   = data_monthly.iloc[-test_n:]

        # ── Lag order selection ───────────────────────────────────────────────
        lag_result = select_order(train, maxlags=CONFIG["vecm"]["max_lags"],
                                   deterministic=CONFIG["vecm"]["deterministic"])
        k_ar_diff = max(1, lag_result.aic - 1)
        logger.info(f"  {self.pair_name}: Selected lag order k_ar_diff={k_ar_diff} (AIC)")

        # ── Cointegration rank selection ─────────────────────────────────────
        rank_result = select_coint_rank(
            train, det_order=0, k_ar_diff=k_ar_diff, signif=0.05
        )
        coint_rank = rank_result.rank
        logger.info(f"  {self.pair_name}: Cointegration rank = {coint_rank}")

        if coint_rank == 0:
            logger.warning(f"  {self.pair_name}: No cointegration detected in sample — fitting VAR instead")
            return {}

        # ── Fit VECM ─────────────────────────────────────────────────────────
        try:
            vecm = VECM(
                train,
                k_ar_diff=k_ar_diff,
                coint_rank=coint_rank,
                deterministic=CONFIG["vecm"]["deterministic"],
                seasons=CONFIG["vecm"]["seasons"],
            )
            fitted = vecm.fit()

            # ── Forecast ─────────────────────────────────────────────────────
            forecast_result = fitted.predict(steps=test_n)
            forecast_df = pd.DataFrame(
                forecast_result,
                index=test.index,
                columns=[base_col, quote_col],
            )

            mape_base  = mape(test[base_col].values, forecast_df[base_col].values)
            mape_quote = mape(test[quote_col].values, forecast_df[quote_col].values)
            rmse_base  = rmse(test[base_col].values, forecast_df[base_col].values)
            rmse_quote = rmse(test[quote_col].values, forecast_df[quote_col].values)

            # ── Adjustment coefficients (alpha) ───────────────────────────────
            # alpha[:,0] = adjustment speed for first cointegrating vector
            alpha = fitted.alpha

            self.results = {
                "pair":             self.pair_name,
                "k_ar_diff":        k_ar_diff,
                "coint_rank":       coint_rank,
                "alpha_base":       round(alpha[0, 0], 6),   # adjustment speed base
                "alpha_quote":      round(alpha[1, 0], 6),   # adjustment speed quote
                "beta":             fitted.beta.tolist(),     # cointegrating vector
                "mape_base_pct":    round(mape_base, 4),
                "mape_quote_pct":   round(mape_quote, 4),
                "rmse_base":        round(rmse_base, 4),
                "rmse_quote":       round(rmse_quote, 4),
                "n_train":          len(train),
                "n_test":           test_n,
                "fitted_model":     fitted,
                "forecast_df":      forecast_df,
                "actual_test":      test,
            }

            # Who adjusts more to correct disequilibrium?
            faster_adjuster = self.base if abs(alpha[0, 0]) > abs(alpha[1, 0]) else self.quote
            logger.info(
                f"  {self.pair_name}: alpha_base={alpha[0,0]:.4f}, alpha_quote={alpha[1,0]:.4f} "
                f"→ {faster_adjuster} adjusts faster (price-follower)"
            )
            logger.info(
                f"  {self.pair_name}: MAPE {self.base}={mape_base:.2f}% "
                f"| {self.quote}={mape_quote:.2f}%"
            )

        except Exception as e:
            logger.error(f"  {self.pair_name}: VECM fitting failed — {e}")
            self.results = {}

        return self.results

    def summary_row(self) -> dict:
        if not self.results:
            return {}
        r = self.results
        return {
            "pair":           r["pair"],
            "k_ar_diff":      r["k_ar_diff"],
            "coint_rank":     r["coint_rank"],
            "alpha_base":     r["alpha_base"],
            "alpha_quote":    r["alpha_quote"],
            "mape_base_pct":  r["mape_base_pct"],
            "mape_quote_pct": r["mape_quote_pct"],
            "rmse_base":      r["rmse_base"],
            "rmse_quote":     r["rmse_quote"],
        }


def run_all_vecm(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run VECM only on confirmed cointegrated pairs."""
    rows = []

    for pair_name, df in tqdm(pairs.items(), desc="VECM modelling"):
        if pair_name not in COINTEGRATED_PAIRS:
            logger.info(f"  Skipping {pair_name} (not in cointegrated set)")
            continue

        model = VECMModel(pair_name, df)
        model.fit()
        row = model.summary_row()
        if row:
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "vecm_results.csv", index=False)
    logger.info(f"\n✅ VECM results saved ({len(results_df)} models)")
    return results_df


def load_pairs() -> dict[str, pd.DataFrame]:
    pairs = {}
    for pair_conf in CONFIG["commodity_pairs"]:
        name = pair_conf["name"]
        path = PROC_DIR / (name.replace("/", "_").lower() + "_merged.csv")
        if path.exists():
            pairs[name] = pd.read_csv(path, index_col=0, parse_dates=True)
    return pairs


def main():
    pairs = load_pairs()
    if not pairs:
        logger.error("No processed pair data found. Run preprocessing.py first.")
        return
    run_all_vecm(pairs)


if __name__ == "__main__":
    main()
