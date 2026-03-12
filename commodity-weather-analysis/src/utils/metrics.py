"""
metrics.py
==========
Evaluation metrics for forecast comparison across all models.
"""

import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask      = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Percentage of correctly predicted price directions (up/down)."""
    actual    = np.array(actual)
    predicted = np.array(predicted)
    actual_dir    = np.sign(np.diff(actual))
    predicted_dir = np.sign(np.diff(predicted))
    return float(np.mean(actual_dir == predicted_dir) * 100)


def theil_u(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Theil's U statistic. 
    U < 1: model beats naive (random walk).
    U = 1: model equals naive.
    U > 1: model worse than naive.
    """
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    naive     = actual[:-1]   # random walk: predict today = yesterday
    actual_h  = actual[1:]
    predicted_h = predicted[1:]
    num   = rmse(actual_h, predicted_h)
    denom = rmse(actual_h, naive)
    return float(num / denom) if denom != 0 else np.nan


def all_metrics(actual: np.ndarray, predicted: np.ndarray, label: str = "") -> dict:
    """Compute all metrics and return as dict."""
    return {
        "model":         label,
        "mape_pct":      round(mape(actual, predicted), 4),
        "rmse":          round(rmse(actual, predicted), 4),
        "mae":           round(mae(actual, predicted), 4),
        "dir_accuracy":  round(directional_accuracy(actual, predicted), 2),
        "theil_u":       round(theil_u(actual, predicted), 4),
    }
