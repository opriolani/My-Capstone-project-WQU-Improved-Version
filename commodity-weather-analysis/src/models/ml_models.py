"""
ml_models.py
============
Machine learning benchmark models: Random Forest, XGBoost, and LSTM.
Used to compare forecast accuracy against SARIMAX/VECM baselines.

Usage:
    python src/models/ml_models.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR    = ROOT / CONFIG["paths"]["processed_data"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results_tables"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMODITIES = list(CONFIG["weather_stations"].keys())
ML_CFG      = CONFIG["ml"]
HORIZON     = ML_CFG["forecast_horizon"]
RS          = ML_CFG["random_state"]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, price_col: str, n_lags: int = 24) -> pd.DataFrame:
    """
    Create supervised learning features from time series:
    - Lagged prices (1..n_lags months)
    - Rolling stats (mean, std)
    - Calendar features (month, quarter)
    - Weather variables (if present)
    """
    monthly = df[[price_col]].resample("ME").last().dropna()

    feat = pd.DataFrame(index=monthly.index)
    feat["price"] = monthly[price_col]

    # Lagged price features
    for lag in range(1, n_lags + 1):
        feat[f"price_lag_{lag}"] = monthly[price_col].shift(lag)

    # Rolling features
    for window in [3, 6, 12]:
        feat[f"rolling_mean_{window}"] = monthly[price_col].shift(1).rolling(window).mean()
        feat[f"rolling_std_{window}"]  = monthly[price_col].shift(1).rolling(window).std()

    # Calendar
    feat["month"]   = feat.index.month
    feat["quarter"] = feat.index.quarter

    # Weather
    if "base_temp" in df.columns:
        feat["temperature"] = df["base_temp"].resample("ME").mean()
    if "base_precip" in df.columns:
        feat["precipitation"] = df["base_precip"].resample("ME").sum()

    return feat.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

def fit_random_forest(X_train, y_train, X_test) -> np.ndarray:
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RS,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def fit_xgboost(X_train, y_train, X_test) -> np.ndarray:
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RS,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model.predict(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# LSTM (TensorFlow / Keras)
# ══════════════════════════════════════════════════════════════════════════════

def fit_lstm(prices: pd.Series) -> dict:
    """
    LSTM on price level (scaled). Uses rolling lookback window.
    Returns dict with predictions and MAPE.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")

        cfg = ML_CFG["lstm"]
        lookback = cfg["lookback"]

        monthly = prices.resample("ME").last().dropna().values.reshape(-1, 1)
        if len(monthly) < lookback + HORIZON + 10:
            return {"mape_pct": np.nan, "rmse": np.nan, "note": "Insufficient data"}

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(monthly)

        def make_sequences(data, lb):
            X, y = [], []
            for i in range(lb, len(data)):
                X.append(data[i - lb:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = make_sequences(scaled, lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = len(X) - HORIZON
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(cfg["units"], return_sequences=True, input_shape=(lookback, 1)),
            Dropout(cfg["dropout"]),
            LSTM(cfg["units"] // 2),
            Dropout(cfg["dropout"]),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=10, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_split=0.1,
            callbacks=[es],
            verbose=0,
        )

        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled).flatten()
        actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mape_val = mean_absolute_percentage_error(actual, pred) * 100
        rmse_val = np.sqrt(np.mean((actual - pred) ** 2))

        return {"mape_pct": round(mape_val, 4), "rmse": round(rmse_val, 4)}

    except ImportError:
        logger.warning("TensorFlow not installed — skipping LSTM")
        return {"mape_pct": np.nan, "rmse": np.nan, "note": "TensorFlow not installed"}
    except Exception as e:
        logger.error(f"LSTM failed: {e}")
        return {"mape_pct": np.nan, "rmse": np.nan, "note": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_ml(
    prices: dict[str, pd.DataFrame],
    pairs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for commodity in tqdm(COMMODITIES, desc="ML benchmarks"):
        if commodity not in prices:
            continue

        # Find matching pair df (for weather features)
        pair_df = None
        for pair_name, pdf in pairs.items():
            base, quote = pair_name.split("/")
            if base == commodity or quote == commodity:
                pair_df = pdf
                break

        price_col = f"{commodity.lower()}_price" if pair_df is not None else "price"
        source_df = pair_df if pair_df is not None and price_col in pair_df.columns else prices[commodity]

        try:
            feat = build_features(source_df, price_col)
            X    = feat.drop(columns=["price"]).values
            y    = feat["price"].values
            split = int(len(X) * (1 - ML_CFG["test_size"]))

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # RF
            rf_pred = fit_random_forest(X_train, y_train, X_test)
            rf_mape = mean_absolute_percentage_error(y_test, rf_pred) * 100

            # XGBoost
            xgb_pred = fit_xgboost(X_train, y_train, X_test)
            xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100

            logger.info(
                f"  {commodity}: RF MAPE={rf_mape:.2f}% | XGB MAPE={xgb_mape:.2f}%"
            )

            row = {
                "commodity":       commodity,
                "rf_mape_pct":     round(rf_mape, 4),
                "xgb_mape_pct":    round(xgb_mape, 4),
                "n_train":         split,
                "n_test":          len(X_test),
            }

        except Exception as e:
            logger.error(f"  {commodity}: ML benchmark failed — {e}")
            row = {"commodity": commodity, "rf_mape_pct": np.nan, "xgb_mape_pct": np.nan}

        # LSTM (separate — needs raw monthly series)
        lstm_res = fit_lstm(prices[commodity]["price"])
        row["lstm_mape_pct"] = lstm_res.get("mape_pct", np.nan)
        row["lstm_rmse"]     = lstm_res.get("rmse", np.nan)

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "ml_benchmark_results.csv", index=False)
    logger.info(f"\n✅ ML benchmark results saved ({len(result_df)} commodities)")
    return result_df


def load_prices() -> dict[str, pd.DataFrame]:
    prices = {}
    for c in COMMODITIES:
        p = PROC_DIR / f"{c.lower()}_prices_processed.csv"
        if p.exists():
            prices[c] = pd.read_csv(p, index_col=0, parse_dates=True)
    return prices


def load_pairs() -> dict[str, pd.DataFrame]:
    pairs = {}
    for pair_conf in CONFIG["commodity_pairs"]:
        name = pair_conf["name"]
        path = PROC_DIR / (name.replace("/", "_").lower() + "_merged.csv")
        if path.exists():
            pairs[name] = pd.read_csv(path, index_col=0, parse_dates=True)
    return pairs


def main():
    prices = load_prices()
    pairs  = load_pairs()
    if not prices:
        logger.error("No processed price data found. Run preprocessing.py first.")
        return
    run_all_ml(prices, pairs)


if __name__ == "__main__":
    main()
