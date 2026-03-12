"""Data fetching modules."""
from .fetch_commodity_data import (
    fetch_all_commodities, load_or_fetch, compute_returns,
    compute_spread, compute_ratio, build_processed_dataset, descriptive_stats
)
from .fetch_weather_data import (
    fetch_all_weather, load_all_weather, align_weather_to_prices
)
