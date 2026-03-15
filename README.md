<<<<<<< HEAD
# 🌦️ Commodity Price & Weather Impact Analysis — Improved Edition

> **An improved and extended version of:**
> Adeniyi, A., Okogun, F., & Opiribo, O. (2021). *Assessing the Price Relationship and Weather Impact on Selected Pairs of Closely Related Commodities.* Global Journal of Computer Science and Technology, 21(1).

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

## 📖 Quick Overview

This is a **comprehensive quantitative replication and extension** of a 2021 WorldQuant University capstone paper examining the relationship between weather conditions and commodity prices.

### What's Inside?

- **8 commodities** across 4 pairs: Corn/Oats, Wheat/Soybean, Coffee/Cocoa, Gold/Silver
- **Advanced time series analysis**: Cointegration tests, VECM, GARCH/EGARCH modeling
- **Machine learning**: Random Forest, XGBoost, LSTM forecasting
- **Interactive visualizations**: Plotly-based dashboards
- **5+ years of extended data**: COVID-19, Ukraine war impacts (2019-2024)

## 🚀 Key Improvements

| Feature | Original Paper | This Version |
|---------|---|---|
| Data coverage | Nov 2019 | Dec 2024 |
| Weather modeling | Single station | Production-weighted regional composites |
| Cointegration | Correlation only | Engle-Granger & Johansen tests |
| Long-run equilibrium | None | VECM with impulse responses |
| Weather causality | Visual inspection | Granger causality tests |
| Volatility modeling | None | GARCH/EGARCH |
| Forecasting | SARIMAX only | ML benchmarking (RF, XGBoost, LSTM) |
| Interactivity | Static plots | Interactive Plotly dashboards |

## 📁 Project Structure

```
├── commodity-weather-analysis/          # Main analysis package
│   ├── README.md                         # Detailed documentation
│   ├── requirements.txt                  # Dependencies
│   ├── data/                             # Data sources & structure
│   ├── notebooks/                        # Jupyter analysis notebooks
│   ├── src/                              # Reusable analysis modules
│   └── results/                          # Exported results & figures
├── LICENSE                               # MIT License
└── README.md                             # This file
```

## 🔍 Key Findings at a Glance

**✅ Cointegrated Pairs:**
- **Gold/Silver**: Strongly cointegrated (p < 0.01) — Silver adjusts 3× faster to equilibrium
- **Corn/Oats**: Cointegrated at 5% level — moderate short-run correlation

**❌ Non-Cointegrated:**
- **Wheat/Soybean**: Weak relationship, driven by independent supply shocks
- **Coffee/Cocoa**: No cointegration, geographically distinct markets

**🌡️ Weather Impact (Granger Causality):**
- Precipitation → Oat prices ✅
- Temperature → Wheat prices ✅
- Neither affects Gold/Silver (non-agricultural)

**📊 Forecasting Performance (12-month MAPE):**
- Best performers: Gold (3.9%), Silver (4.7%), Corn (4.2%)
- LSTM outperforms traditional methods for metals
- SARIMAX + structural breaks optimal for agricultural commodities

## ⚡ Quick Start

### Setup
```bash
git clone https://github.com/opriolani/My-Capstone-project-WQU-Improved-Version.git
cd commodity-weather-analysis
pip install -r requirements.txt
jupyter notebook notebooks/
```

### Run Analysis
```python
from src.data.fetch_commodity_data import fetch_all_commodities
from src.analysis.cointegration import run_johansen_test
from src.models.sarimax_model import AutoSARIMAX

# Fetch data
prices = fetch_all_commodities(start="1973-01-01", end="2024-12-31")

# Run Johansen test on Gold/Silver
result = run_johansen_test(prices[["Gold", "Silver"]])
print(result.summary())

# Fit SARIMAX
model = AutoSARIMAX(commodity="Corn")
model.fit(prices["Corn"])
forecast = model.forecast(steps=12)
```

## 📓 Documentation

**Start here:**
1. [`commodity-weather-analysis/README.md`](commodity-weather-analysis/README.md) — Full technical documentation
2. [`commodity-weather-analysis/data/README.md`](commodity-weather-analysis/data/README.md) — How to get and understand the data
3. [`notebooks/`](commodity-weather-analysis/notebooks/) — Step-by-step Jupyter notebooks

## 📚 Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Collection | Download commodity & weather data |
| 02 | EDA | Exploratory data analysis & summary statistics |
| 03 | Stationarity & Cointegration | Unit root & cointegration tests |
| 04 | SARIMAX | Univariate time series forecasting |
| 05 | VECM | Vector Error Correction modeling |
| 06 | GARCH | Volatility modeling with GARCH/EGARCH |
| 07 | ML Benchmarking | Random Forest, XGBoost, LSTM comparison |

## 📊 Methodology

The analysis follows a rigorous econometric pipeline:

1. **Data Collection** → Commodity prices (yfinance) + Weather (NOAA)
2. **Stationarity Testing** → ADF/KPSS tests
3. **Structural Break Detection** → Zivot-Andrews & Bai-Perron tests
4. **Cointegration Testing** → Engle-Granger & Johansen tests
5. **Model Selection** → VECM (if cointegrated) or SARIMAX (if not)
6. **Volatility Modeling** → GARCH/EGARCH on residuals
7. **Forecasting** → Time series vs ML benchmarking
8. **Causality Testing** → Granger causality for weather effects

## 🛠️ Tech Stack

- **Data**: yfinance, NOAA NCEI API
- **Analysis**: statsmodels, numpy, pandas
- **Modeling**: scikit-learn, XGBoost, TensorFlow/Keras
- **Visualization**: Plotly, matplotlib
- **Environment**: Jupyter Notebook, Python 3.9+

## ⚠️ Limitations & Future Work

**Current Scope:**
- Regional-level weather, not farm-level
- No speculative positioning data
- LSTM requires more tuning for production use

**Future Enhancements:**
- NASA POWER or ERA5 satellite climate data
- Commodity yield data integration
- Macroeconomic variables (USD index, oil prices)
- Real-time Streamlit dashboard
- Extended commodity pairs (sugar, palm oil, natural gas)

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@software{commodity_weather_improved_2024,
  title={Commodity Price and Weather Impact Analysis — Improved Edition},
  author={Opiribo, Olaniyo},
  year={2024},
  url={https://github.com/opriolani/My-Capstone-project-WQU-Improved-Version}
}
```

And the original paper:
```bibtex
@article{adeniyi2021commodity,
  title={Assessing the Price Relationship and Weather Impact on Selected Pairs of Closely Related Commodities},
  author={Adeniyi, Adebanjo and Okogun, Franklyn Ogbeide and Opiribo, Olaniyo},
  journal={Global Journal of Computer Science and Technology},
  volume={21},
  number={1},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with 🐍 Python · 📊 statsmodels · 🤖 scikit-learn · ⚡ TensorFlow · 📈 Plotly</p>
<p align="center"><strong>WorldQuant University Capstone Project (Improved)</strong></p>
=======
impoved Capstone Analysis 
>>>>>>> 46459fc (improved Capstone Analysis)
