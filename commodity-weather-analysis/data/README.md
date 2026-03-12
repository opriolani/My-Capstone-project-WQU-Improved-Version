# Data

Raw data files are **not tracked** in this repository due to licensing restrictions.

## How to Obtain the Data

### 1. Commodity Prices (via yfinance — free)
Run the data collection script, which downloads from Yahoo Finance:
```bash
python src/data_collection.py
```

### 2. Weather Data (NOAA CDO — free API token required)
1. Register for a free token at: https://www.ncdc.noaa.gov/cdo-web/token
2. Export your token:
   ```bash
   export NOAA_TOKEN=your_token_here
   ```
3. Re-run data collection — weather will be fetched automatically.

## Directory Structure After Download
```
data/
├── raw/
│   ├── prices/
│   │   ├── corn_prices.csv
│   │   ├── oats_prices.csv
│   │   └── ... (8 files)
│   └── weather/
│       ├── corn_weather.csv
│       └── ... (8 files)
└── processed/
    ├── corn_prices_processed.csv
    ├── corn_oats_merged.csv
    └── ...
```

## Original Paper Data Sources
- Macrotrends: https://www.macrotrends.net/
- Quandl/Nasdaq Data Link: https://data.nasdaq.com/
- NOAA CDO: https://www.ncdc.noaa.gov/cdo-web/
