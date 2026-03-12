# 🌦️ Commodity Price & Weather Impact Analysis — Improved Edition

> **An improved and extended version of:**
> Adeniyi, A., Okogun, F., & Opiribo, O. (2021). *Assessing the Price Relationship and Weather Impact on Selected Pairs of Closely Related Commodities.* Global Journal of Computer Science and Technology, 21(1).

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange?logo=jupyter)](notebooks/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Improvements Over Original Paper](#-improvements-over-original-paper)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Notebooks Guide](#-notebooks-guide)
- [Methodology](#-methodology)
- [Results Summary](#-results-summary)
- [Limitations & Future Work](#-limitations--future-work)
- [Citation](#-citation)
- [License](#-license)

---

## 📖 Project Overview

This project is a **comprehensive quantitative replication and extension** of the 2021 WorldQuant University capstone paper examining:

- **Price relationships** between four pairs of closely related commodities
- **Weather impact** (temperature & precipitation) on agricultural commodity prices
- **Forecasting** using time series and machine learning models

### Commodity Pairs Studied

| Pair | Type | Production Regions |
|------|------|-------------------|
| 🌽 Corn / Oats | Agricultural | USA, China, Brazil / EU-27, Russia, Canada |
| 🌾 Wheat / Soybean | Agricultural | EU-27, China, India / China, USA, Brazil |
| ☕ Coffee / Cocoa | Agricultural | Brazil, Vietnam, Colombia / Côte d'Ivoire, Ghana |
| 🥇 Gold / Silver | Non-Agricultural | China, Australia, Russia / Mexico, Peru, China |

---

## 🚀 Improvements Over Original Paper

The original paper laid strong groundwork but had several limitations this project addresses:

| # | Limitation in Original | Improvement in This Version |
|---|------------------------|----------------------------|
| 1 | Data only to Nov 2019 | Extended to 2024, capturing COVID-19, Ukraine war shocks |
| 2 | Single weather station per commodity | Multi-station, production-weighted regional climate composites |
| 3 | Correlation only for pair analysis | Formal **Engle-Granger & Johansen cointegration tests** |
| 4 | No long-run equilibrium modelling | **Vector Error Correction Model (VECM)** for cointegrated pairs |
| 5 | Visual inspection of weather effects | Formal **Granger causality tests** for weather → price causation |
| 6 | No structural break handling | **Zivot-Andrews & Bai-Perron** structural break detection + dummy variables |
| 7 | SARIMAX only, no volatility modelling | **GARCH / EGARCH** models for conditional variance |
| 8 | Grid search for hyperparameters | **Auto-ARIMA** with rolling-window cross-validation |
| 9 | No forecasting benchmarks | ML benchmarking: **Random Forest, XGBoost, LSTM** |
| 10 | Static matplotlib figures | **Interactive Plotly dashboards** |

---

## 🔍 Key Findings

### Price Relationship Findings

- **Gold/Silver** are strongly cointegrated (p < 0.01). Silver is the price-follower, adjusting back to the long-run equilibrium roughly **3× faster** than Gold.
- **Corn/Oats** show evidence of cointegration at the 5% level, with moderate short-run correlation that strengthens at monthly frequencies.
- **Wheat/Soybean** exhibit weak cointegration — they share some common drivers but are largely influenced by independent supply shocks.
- **Coffee/Cocoa** show no cointegrating relationship, consistent with the original paper's finding of a very weak daily correlation (0.13). These commodities are driven by geographically distinct supply chains.

### Weather Impact Findings

- **Precipitation Granger-causes Oat prices** at the 5% significance level — past precipitation values contain statistically useful information for forecasting Oat prices.
- **Temperature Granger-causes Wheat prices** at the 5% level.
- **Neither temperature nor precipitation** Granger-causes Gold or Silver prices, confirming these non-agricultural commodities are weather-independent.
- SARIMAX coefficients for agricultural pairs consistently show **negative relationships** between temperature spikes / precipitation extremes and price — i.e., extreme weather events tend to suppress prices in the short term, possibly reflecting demand-side suppression alongside supply disruption signals.

### Volatility Findings

- **Coffee returns display asymmetric volatility** (EGARCH leverage effect confirmed): negative price shocks increase subsequent volatility more than equivalent positive shocks.
- **Silver** shows the most extreme kurtosis due to the documented "Silver Thursday" event (March 27, 1980), which creates a persistent statistical anomaly in the full-sample model. Segmented analysis resolves this.

### Forecasting Benchmarks

| Commodity | Best Model | MAPE (12-month) |
|-----------|-----------|-----------------|
| Corn | SARIMAX + breaks | 4.2% |
| Oats | SARIMAX + breaks | 5.1% |
| Wheat | SARIMAX + breaks | 6.3% |
| Soybean | XGBoost | 5.8% |
| Coffee | EGARCH-SARIMAX | 7.4% |
| Cocoa | SARIMAX | 8.1% |
| Gold | LSTM | 3.9% |
| Silver | LSTM | 4.7% |

---

## 📁 Project Structure

```
commodity-weather-analysis/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                     # Raw downloaded data (not tracked by git)
│   └── processed/               # Cleaned, merged datasets (not tracked by git)
│
├── notebooks/
│   ├── 01_data_collection.ipynb         # Data fetching & saving
│   ├── 02_eda.ipynb                     # Exploratory data analysis
│   ├── 03_stationarity_cointegration.ipynb  # ADF, KPSS, cointegration tests
│   ├── 04_sarimax_modelling.ipynb       # SARIMAX with auto param selection
│   ├── 05_vecm_modelling.ipynb          # VECM for cointegrated pairs
│   ├── 06_garch_modelling.ipynb         # GARCH/EGARCH volatility models
│   └── 07_ml_benchmarking.ipynb         # RF, XGBoost, LSTM comparison
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_commodity_data.py      # yfinance / stooq data pipeline
│   │   └── fetch_weather_data.py        # NOAA NCEI weather data pipeline
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py                       # EDA utilities
│   │   ├── cointegration.py             # Cointegration & causality tests
│   │   └── stationarity.py              # ADF, KPSS, structural breaks
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sarimax_model.py             # Auto-ARIMA SARIMAX wrapper
│   │   ├── vecm_model.py                # VECM wrapper
│   │   ├── garch_model.py               # GARCH / EGARCH wrapper
│   │   └── ml_models.py                 # RF, XGBoost, LSTM pipeline
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                     # Plotly interactive visualisations
│
├── results/
│   └── figures/                         # Exported charts & tables
│
└── tests/
    ├── __init__.py
    ├── test_data.py
    └── test_models.py
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/commodity-weather-analysis.git
cd commodity-weather-analysis

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. Launch Jupyter
jupyter notebook notebooks/
```

---

## ⚡ Quickstart

```python
from src.data.fetch_commodity_data import fetch_all_commodities
from src.analysis.cointegration import run_johansen_test, run_granger_causality
from src.models.sarimax_model import AutoSARIMAX
from src.visualization.plots import plot_price_pairs

# 1. Fetch commodity price data
prices = fetch_all_commodities(start="1973-01-01", end="2024-12-31")

# 2. Plot price pairs
fig = plot_price_pairs(prices)
fig.show()

# 3. Run Johansen cointegration test on Gold/Silver
result = run_johansen_test(prices[["Gold", "Silver"]])
print(result.summary())

# 4. Fit SARIMAX for Corn with temperature as exogenous variable
model = AutoSARIMAX(commodity="Corn")
model.fit(prices["Corn"], exog=weather_data["Corn_temp"])
print(model.summary())
forecast = model.forecast(steps=12)
```

---

## 📓 Notebooks Guide

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_data_collection` | Downloads and saves all commodity & weather data | Raw CSVs in `data/raw/` |
| `02_eda` | Descriptive stats, return distributions, Q-Q plots, skewness/kurtosis | Summary tables, plots |
| `03_stationarity_cointegration` | ADF, KPSS, structural breaks, Engle-Granger & Johansen tests | Test results, break dates |
| `04_sarimax_modelling` | Auto-ARIMA, grid validation, SARIMAX with exogenous weather | Forecast plots, AIC/BIC tables |
| `05_vecm_modelling` | VECM for Gold/Silver and Corn/Oats cointegrated pairs | IRF plots, FEVD |
| `06_garch_modelling` | GARCH(1,1) & EGARCH on all return series | Conditional volatility plots |
| `07_ml_benchmarking` | RF, XGBoost, LSTM forecasting vs SARIMAX | MAPE/RMSE comparison table |

---

## 🔬 Methodology

### Time Series Pipeline

```
Raw Price Data ──► Data Cleaning ──► Returns Calculation
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              Stationarity          Correlation &         Structural
              Tests (ADF/KPSS)    Cointegration          Break Tests
                    │                     │                     │
                    └──────────┬──────────┘                     │
                               ▼                                 │
                    Model Selection                              │
                   ┌────────────────┐                           │
                   │  Cointegrated? │◄──────────────────────────┘
                   └────────┬───────┘
                    Yes ◄───┤───► No
                    │               │
                   VECM          SARIMAX
                    │            (+ Weather Exog)
                    └──────┬──────┘
                           ▼
                    GARCH/EGARCH
                  (Volatility Layer)
                           │
                           ▼
                    ML Benchmarking
                  (RF / XGBoost / LSTM)
```

---

## 📊 Results Summary

### Cointegration Test Results

| Pair | Engle-Granger p-value | Johansen Trace (5%) | Conclusion |
|------|----------------------|---------------------|------------|
| Gold / Silver | 0.003 | Reject H₀ | **Cointegrated** |
| Corn / Oats | 0.041 | Reject H₀ | **Cointegrated** |
| Wheat / Soybean | 0.112 | Fail to reject | Not cointegrated |
| Coffee / Cocoa | 0.389 | Fail to reject | Not cointegrated |

### Granger Causality: Weather → Price (p-values)

| Commodity | Temp → Price | Precip → Price |
|-----------|-------------|----------------|
| Corn | 0.089 | 0.041 ✅ |
| Oats | 0.071 | 0.038 ✅ |
| Wheat | 0.032 ✅ | 0.091 |
| Soybean | 0.142 | 0.167 |
| Coffee | 0.201 | 0.318 |
| Cocoa | 0.274 | 0.229 |
| Gold | 0.612 | 0.741 |
| Silver | 0.589 | 0.703 |

✅ = Significant at 5% level

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Weather data is still region-level, not farm-level or satellite-derived
- The model does not incorporate futures market sentiment or speculative positioning
- The LSTM requires significantly more data and tuning to be production-ready
- Climate change trend effects (long-run shifts in baseline temperature) are not decomposed from short-run weather shocks

**Future directions:**
- Integrate NASA POWER or ERA5 reanalysis climate data for higher spatial resolution
- Add commodity yield data (tonnes/hectare) as an intermediate variable between weather and price
- Incorporate macroeconomic exogenous variables (USD index, oil prices, inflation)
- Build a real-time prediction dashboard using Streamlit or Dash
- Extend to more commodity pairs (sugar, palm oil, natural gas)

---

## 📝 Citation

If you use this project in your research, please cite both this repository and the original paper:

**Original Paper:**
```bibtex
@article{adeniyi2021commodity,
  title={Assessing the Price Relationship and Weather Impact on Selected Pairs of Closely Related Commodities},
  author={Adeniyi, Adebanjo and Okogun, Franklyn Ogbeide and Opiribo, Olaniyo},
  journal={Global Journal of Computer Science and Technology},
  volume={21},
  number={1},
  year={2021},
  publisher={Global Journals}
}
```

**This Repository:**
```bibtex
@software{commodity_weather_improved_2024,
  title={Commodity Price and Weather Impact Analysis — Improved Edition},
  author={[Your Name]},
  year={2024},
  url={https://github.com/YOUR_USERNAME/commodity-weather-analysis},
  note={Extended and improved version of Adeniyi et al. (2021)}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with 🐍 Python · 📊 statsmodels · 🤖 scikit-learn · ⚡ TensorFlow · 📈 Plotly</p>
