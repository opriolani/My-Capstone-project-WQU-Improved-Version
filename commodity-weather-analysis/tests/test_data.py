"""
test_data.py
------------
Unit tests for data fetching and preprocessing utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '..')


@pytest.fixture
def sample_prices():
    """Small synthetic price DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "Gold":   100 + np.cumsum(np.random.randn(100) * 0.5),
        "Silver": 20  + np.cumsum(np.random.randn(100) * 0.2),
        "Corn":   80  + np.cumsum(np.random.randn(100) * 0.4),
    }, index=dates)


class TestDataProcessing:

    def test_compute_daily_returns(self, sample_prices):
        from src.data.fetch_commodity_data import compute_returns
        ret = compute_returns(sample_prices, freq="D")
        assert isinstance(ret, pd.DataFrame)
        assert len(ret) == len(sample_prices) - 1
        assert list(ret.columns) == list(sample_prices.columns)

    def test_compute_weekly_returns(self, sample_prices):
        from src.data.fetch_commodity_data import compute_returns
        ret = compute_returns(sample_prices, freq="W")
        assert isinstance(ret, pd.DataFrame)
        assert len(ret) < len(sample_prices)

    def test_compute_spread(self, sample_prices):
        from src.data.fetch_commodity_data import compute_spread
        spread = compute_spread(sample_prices, ("Gold", "Silver"))
        assert isinstance(spread, pd.Series)
        assert spread.name == "Gold_minus_Silver"
        assert len(spread) == len(sample_prices)

    def test_compute_ratio(self, sample_prices):
        from src.data.fetch_commodity_data import compute_ratio
        ratio = compute_ratio(sample_prices, ("Gold", "Silver"))
        assert isinstance(ratio, pd.Series)
        assert ratio.name == "Gold_over_Silver"
        # All values should be positive
        assert (ratio > 0).all()

    def test_descriptive_stats(self, sample_prices):
        from src.data.fetch_commodity_data import descriptive_stats
        stats = descriptive_stats(sample_prices)
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert "skewness" in stats.columns
        assert "kurtosis" in stats.columns
        assert len(stats) == len(sample_prices.columns)

    def test_build_processed_dataset(self, sample_prices, tmp_path):
        from src.data.fetch_commodity_data import build_processed_dataset
        dataset = build_processed_dataset(sample_prices, save=False)
        assert "prices" in dataset
        assert "daily_returns" in dataset
        assert "weekly_returns" in dataset
        assert "monthly_returns" in dataset
        assert "spreads" in dataset
        assert "ratios" in dataset

    def test_missing_value_interpolation(self):
        """Test that cleaning handles NaN gaps correctly."""
        from src.data.fetch_commodity_data import _clean_prices
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"A": [1.0, np.nan, np.nan, 4.0, 5.0,
                                      6.0, np.nan, 8.0, 9.0, 10.0]}, index=dates)
        cleaned = _clean_prices(prices)
        assert cleaned.isna().sum().sum() == 0
        # Interpolated values should be between surrounding valid values
        assert 1.0 < cleaned["A"].iloc[1] < 4.0
