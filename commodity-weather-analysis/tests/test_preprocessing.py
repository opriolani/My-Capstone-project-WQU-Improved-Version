"""
tests/test_preprocessing.py
============================
Unit tests for preprocessing functions.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.metrics import mape, rmse, mae, directional_accuracy, theil_u


# ── Metrics Tests ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_mape_perfect_forecast(self):
        actual = np.array([100, 200, 300])
        assert mape(actual, actual) == pytest.approx(0.0)

    def test_mape_known_value(self):
        actual    = np.array([100, 100, 100])
        predicted = np.array([110, 90, 100])
        # errors: 10%, 10%, 0% → mean = 6.67%
        assert mape(actual, predicted) == pytest.approx(6.6667, rel=1e-3)

    def test_mape_ignores_zero_actual(self):
        actual    = np.array([0, 100, 200])
        predicted = np.array([10, 110, 210])
        # Only non-zero actuals: errors 10%, 5% → mean 7.5%
        assert mape(actual, predicted) == pytest.approx(7.5, rel=1e-3)

    def test_rmse_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 2.0, 2.0])
        # errors: 1, 0, 1 → sqrt(mean(1,0,1)) = sqrt(2/3)
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(2 / 3), rel=1e-5)

    def test_mae_known_value(self):
        actual    = np.array([10, 20, 30])
        predicted = np.array([12, 18, 33])
        assert mae(actual, predicted) == pytest.approx(2.333, rel=1e-3)

    def test_directional_accuracy_perfect(self):
        actual    = np.array([1.0, 2.0, 3.0, 2.5])
        predicted = np.array([1.1, 2.2, 3.1, 2.4])
        # Directions: +, +, - / +, +, - → 100%
        assert directional_accuracy(actual, predicted) == pytest.approx(100.0)

    def test_directional_accuracy_worst(self):
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 0.5, 0.1])
        # actual: +, + / predicted: -, - → 0%
        assert directional_accuracy(actual, predicted) == pytest.approx(0.0)

    def test_theil_u_better_than_naive(self):
        # Perfect forecast should give U ≈ 0
        actual    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual.copy()
        u = theil_u(actual, predicted)
        assert u < 1.0


# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:

    def make_price_series(self, n: int = 100) -> pd.Series:
        idx = pd.date_range("2000-01-01", periods=n, freq="D")
        vals = 100 + np.random.randn(n).cumsum()
        return pd.Series(vals, index=idx, name="price")

    def test_returns_calculation(self):
        prices = self.make_price_series()
        returns = prices.pct_change().dropna()
        assert len(returns) == len(prices) - 1
        assert not returns.isna().any()

    def test_interpolation_fills_gaps(self):
        prices = self.make_price_series(50)
        prices.iloc[10:15] = np.nan  # introduce gaps
        filled = prices.interpolate(method="time")
        assert not filled.isna().any()

    def test_log_returns_shape(self):
        prices = self.make_price_series()
        log_ret = np.log(prices).diff().dropna()
        assert len(log_ret) == len(prices) - 1

    def test_rolling_vol(self):
        prices = self.make_price_series()
        returns = prices.pct_change()
        vol = returns.rolling(21).std() * np.sqrt(252)
        # First 21 values should be NaN
        assert vol.iloc[:20].isna().all()
        assert not vol.iloc[21:].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
