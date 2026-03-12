"""
test_models.py
--------------
Unit tests for all modelling modules.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '..')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """Synthetic commodity price series for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    gold   = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates, name="Gold")
    silver = pd.Series( 20 + np.cumsum(np.random.randn(n) * 0.2), index=dates, name="Silver")
    corn   = pd.Series( 80 + np.cumsum(np.random.randn(n) * 0.4), index=dates, name="Corn")
    oats   = pd.Series( 60 + np.cumsum(np.random.randn(n) * 0.3), index=dates, name="Oats")
    return pd.DataFrame({"Gold": gold, "Silver": silver, "Corn": corn, "Oats": oats})


@pytest.fixture
def synthetic_returns(synthetic_prices):
    return synthetic_prices.pct_change().dropna()


@pytest.fixture
def synthetic_weather(synthetic_prices):
    np.random.seed(99)
    n = len(synthetic_prices)
    return pd.DataFrame({
        "TAVG": np.random.randn(n) * 5 + 20,
        "PRCP": np.abs(np.random.randn(n) * 2),
    }, index=synthetic_prices.index)


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

class TestStationarity:

    def test_adf_on_random_walk_is_nonstationary(self):
        from src.analysis.stationarity import adf_test
        np.random.seed(1)
        rw = pd.Series(np.cumsum(np.random.randn(500)), name="RW")
        result = adf_test(rw)
        assert result["test"] == "ADF"
        assert result["conclusion"] == "Non-Stationary"

    def test_adf_on_white_noise_is_stationary(self):
        from src.analysis.stationarity import adf_test
        np.random.seed(2)
        wn = pd.Series(np.random.randn(500), name="WN")
        result = adf_test(wn)
        assert result["conclusion"] == "Stationary"

    def test_kpss_on_white_noise_is_stationary(self):
        from src.analysis.stationarity import kpss_test
        np.random.seed(3)
        wn = pd.Series(np.random.randn(500), name="WN")
        result = kpss_test(wn)
        assert result["test"] == "KPSS"
        assert "conclusion" in result

    def test_bai_perron_returns_dict(self, synthetic_prices):
        from src.analysis.stationarity import bai_perron_breaks
        result = bai_perron_breaks(synthetic_prices["Gold"], max_breaks=3)
        assert "n_breaks" in result
        assert "break_dates" in result
        assert "segments" in result
        assert isinstance(result["segments"], list)

    def test_full_stationarity_report_shape(self, synthetic_prices):
        from src.analysis.stationarity import full_stationarity_report
        report = full_stationarity_report(synthetic_prices)
        assert len(report) == len(synthetic_prices.columns)
        assert "ADF_conclusion" in report.columns
        assert "KPSS_conclusion" in report.columns


# ---------------------------------------------------------------------------
# Cointegration tests
# ---------------------------------------------------------------------------

class TestCointegration:

    def test_engle_granger_returns_dict(self, synthetic_prices):
        from src.analysis.cointegration import engle_granger_test
        result = engle_granger_test(synthetic_prices["Gold"], synthetic_prices["Silver"])
        assert "p_value" in result
        assert "hedge_ratio" in result
        assert "cointegrated" in result
        assert isinstance(result["cointegrated"], bool)

    def test_johansen_returns_rank(self, synthetic_prices):
        from src.analysis.cointegration import johansen_test
        result = johansen_test(synthetic_prices[["Gold", "Silver"]])
        assert "cointegration_rank" in result
        assert result["cointegration_rank"] in [0, 1, 2]

    def test_correlation_report_shape(self, synthetic_returns):
        from src.analysis.cointegration import correlation_report
        corr = correlation_report(synthetic_returns)
        assert corr.shape == (len(synthetic_returns.columns), len(synthetic_returns.columns))
        # Diagonal should be 1.0
        for col in synthetic_returns.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-9

    def test_granger_causality_runs(self, synthetic_returns, synthetic_weather):
        from src.analysis.cointegration import granger_causality_weather
        result = granger_causality_weather(
            synthetic_returns["Gold"],
            synthetic_weather["TAVG"].rename("Gold_TAVG"),
            max_lags=5,
        )
        assert "causality_found" in result
        assert "min_pvalue" in result
        assert 0.0 <= result["min_pvalue"] <= 1.0


# ---------------------------------------------------------------------------
# SARIMAX model tests
# ---------------------------------------------------------------------------

class TestSARIMAX:

    def test_sarimax_fits_without_exog(self, synthetic_prices):
        from src.models.sarimax_model import AutoSARIMAX
        series = synthetic_prices["Gold"].iloc[:200]
        model  = AutoSARIMAX("Gold", seasonal_period=12)
        model.fit(series)
        assert model.result_ is not None
        assert model.order_ is not None

    def test_sarimax_forecast_returns_dataframe(self, synthetic_prices):
        from src.models.sarimax_model import AutoSARIMAX
        series = synthetic_prices["Gold"].iloc[:200]
        model  = AutoSARIMAX("Gold", seasonal_period=12)
        model.fit(series)
        fc = model.forecast(steps=6)
        assert isinstance(fc, pd.DataFrame)
        assert "forecast" in fc.columns
        assert len(fc) == 6

    def test_sarimax_diagnostics(self, synthetic_prices):
        from src.models.sarimax_model import AutoSARIMAX
        series = synthetic_prices["Gold"].iloc[:200]
        model  = AutoSARIMAX("Gold", seasonal_period=12)
        model.fit(series)
        d = model.diagnostics()
        assert "aic" in d
        assert "bic" in d
        assert "log_likelihood" in d

    def test_sarimax_with_exog(self, synthetic_prices, synthetic_weather):
        from src.models.sarimax_model import AutoSARIMAX
        series = synthetic_prices["Gold"].iloc[:200]
        weather_subset = synthetic_weather.iloc[:200]
        model = AutoSARIMAX("Gold", seasonal_period=12)
        model.fit(series, exog=weather_subset)
        assert model.result_ is not None


# ---------------------------------------------------------------------------
# VECM model tests
# ---------------------------------------------------------------------------

class TestVECM:

    def test_vecm_fits(self, synthetic_prices):
        from src.models.vecm_model import VECMModel
        vm = VECMModel(pair=("Gold", "Silver"), coint_rank=1, k_ar_diff=2)
        vm.fit(synthetic_prices[["Gold", "Silver"]])
        assert vm.result_ is not None

    def test_vecm_forecast_shape(self, synthetic_prices):
        from src.models.vecm_model import VECMModel
        vm = VECMModel(pair=("Gold", "Silver"), coint_rank=1, k_ar_diff=2)
        vm.fit(synthetic_prices[["Gold", "Silver"]])
        fc = vm.forecast(steps=5)
        assert isinstance(fc, pd.DataFrame)
        assert len(fc) == 5

    def test_vecm_adjustment_coefficients(self, synthetic_prices):
        from src.models.vecm_model import VECMModel
        vm = VECMModel(pair=("Gold", "Silver"), coint_rank=1, k_ar_diff=2)
        vm.fit(synthetic_prices[["Gold", "Silver"]])
        alpha = vm.adjustment_coefficients()
        assert alpha.shape == (2, 1)


# ---------------------------------------------------------------------------
# GARCH model tests
# ---------------------------------------------------------------------------

class TestGARCH:

    def test_garch_fits(self, synthetic_returns):
        from src.models.garch_model import CommodityGARCH
        g = CommodityGARCH("Gold", model_type="GARCH")
        g.fit(synthetic_returns["Gold"])
        assert g.result_ is not None

    def test_egarch_fits(self, synthetic_returns):
        from src.models.garch_model import CommodityGARCH
        g = CommodityGARCH("Gold", model_type="EGARCH")
        g.fit(synthetic_returns["Gold"])
        assert g.result_ is not None

    def test_conditional_volatility_shape(self, synthetic_returns):
        from src.models.garch_model import CommodityGARCH
        g = CommodityGARCH("Gold", model_type="GARCH")
        g.fit(synthetic_returns["Gold"])
        vol = g.conditional_volatility()
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(synthetic_returns["Gold"])

    def test_leverage_effect_egarch(self, synthetic_returns):
        from src.models.garch_model import CommodityGARCH
        g = CommodityGARCH("Gold", model_type="EGARCH")
        g.fit(synthetic_returns["Gold"])
        lev = g.leverage_effect()
        assert "leverage_effect" in lev
        assert isinstance(lev["leverage_effect"], bool)

    def test_leverage_effect_not_applicable_for_garch(self, synthetic_returns):
        from src.models.garch_model import CommodityGARCH
        g = CommodityGARCH("Gold", model_type="GARCH")
        g.fit(synthetic_returns["Gold"])
        lev = g.leverage_effect()
        assert lev["applicable"] is False


# ---------------------------------------------------------------------------
# ML models tests
# ---------------------------------------------------------------------------

class TestMLModels:

    def test_feature_engineering(self, synthetic_prices, synthetic_weather):
        from src.models.ml_models import build_features, split_features_target
        feat = build_features(synthetic_prices["Gold"], weather=synthetic_weather)
        assert "target" in feat.columns
        X, y = split_features_target(feat)
        assert len(X) == len(y)
        assert "target" not in X.columns

    def test_random_forest_fits_and_predicts(self, synthetic_prices, synthetic_weather):
        from src.models.ml_models import (
            RandomForestForecaster, build_features, split_features_target
        )
        feat = build_features(synthetic_prices["Gold"].iloc[:400],
                              weather=synthetic_weather.iloc[:400])
        X, y = split_features_target(feat)
        train_X, test_X = X.iloc[:350], X.iloc[350:]
        train_y, test_y = y.iloc[:350], y.iloc[350:]

        rf = RandomForestForecaster("Gold", n_estimators=50)
        rf.fit(train_X, train_y)
        preds = rf.predict(test_X)
        assert len(preds) == len(test_y)
        assert not np.any(np.isnan(preds))

    def test_feature_importance_returns_series(self, synthetic_prices, synthetic_weather):
        from src.models.ml_models import (
            RandomForestForecaster, build_features, split_features_target
        )
        feat = build_features(synthetic_prices["Gold"].iloc[:400],
                              weather=synthetic_weather.iloc[:400])
        X, y = split_features_target(feat)
        rf = RandomForestForecaster("Gold", n_estimators=50)
        rf.fit(X, y)
        imp = rf.feature_importance()
        assert isinstance(imp, pd.Series)
        assert abs(imp.sum() - 1.0) < 1e-6  # Importances sum to 1
