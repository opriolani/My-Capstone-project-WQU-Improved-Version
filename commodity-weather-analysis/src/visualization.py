"""
visualization.py
================
Static (matplotlib/seaborn) and interactive (plotly) visualizations.
Covers all key charts from both the original paper and new improvements.

Usage:
    python src/visualization.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

PROC_DIR    = ROOT / CONFIG["paths"]["processed_data"]
RESULTS_DIR = ROOT / CONFIG["paths"]["results_tables"]
FIG_DIR     = ROOT / CONFIG["paths"]["results_figures"]
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
PAIR_COLORS = {"Corn/Oats": "#2196F3", "Wheat/Soybean": "#4CAF50",
               "Coffee/Cocoa": "#FF9800", "Gold/Silver": "#9C27B0"}


# ══════════════════════════════════════════════════════════════════════════════
# STATIC CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_pair_prices(pair_name: str, df: pd.DataFrame) -> None:
    """Plot price levels for both commodities in a pair on a dual-axis chart."""
    base, quote = pair_name.split("/")
    b_col = f"{base.lower()}_price"
    q_col = f"{quote.lower()}_price"

    if b_col not in df.columns or q_col not in df.columns:
        return

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.plot(df.index, df[b_col], color="#1565C0", lw=1.2, alpha=0.85, label=base)
    ax2.plot(df.index, df[q_col], color="#B71C1C", lw=1.2, alpha=0.85, label=quote)

    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{base} Price", color="#1565C0")
    ax2.set_ylabel(f"{quote} Price", color="#B71C1C")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#B71C1C")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    plt.title(f"{pair_name} — Historical Prices", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = FIG_DIR / f"{pair_name.replace('/', '_').lower()}_prices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_returns_distribution(pair_name: str, df: pd.DataFrame) -> None:
    """QQ plot + histogram of daily returns for each commodity in a pair."""
    from scipy import stats

    base, quote = pair_name.split("/")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{pair_name} — Return Distributions", fontsize=14, fontweight="bold")

    for idx, (commodity, ax_hist, ax_qq) in enumerate([
        (base,  axes[0][0], axes[0][1]),
        (quote, axes[1][0], axes[1][1]),
    ]):
        ret_col = f"{commodity.lower()}_daily_return"
        if ret_col not in df.columns:
            continue

        returns = df[ret_col].dropna()

        # Histogram
        ax_hist.hist(returns, bins=80, color=PAIR_COLORS.get(pair_name, "#607D8B"),
                     edgecolor="white", alpha=0.8, density=True)
        xr = np.linspace(returns.min(), returns.max(), 200)
        ax_hist.plot(xr, stats.norm.pdf(xr, returns.mean(), returns.std()),
                     "r-", lw=2, label="Normal fit")
        ax_hist.set_title(f"{commodity} Daily Returns")
        ax_hist.set_xlabel("Return")
        ax_hist.legend()

        # QQ Plot
        (osm, osr), (slope, intercept, _) = stats.probplot(returns, dist="norm")
        ax_qq.scatter(osm, osr, s=4, alpha=0.4, color=PAIR_COLORS.get(pair_name, "#607D8B"))
        ax_qq.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
        ax_qq.set_title(f"{commodity} Q-Q Plot")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")

    plt.tight_layout()
    out = FIG_DIR / f"{pair_name.replace('/', '_').lower()}_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_correlation_heatmap(pairs: dict[str, pd.DataFrame]) -> None:
    """Correlation heatmap across all 8 commodity daily returns."""
    returns = {}
    for pair_name, df in pairs.items():
        base, quote = pair_name.split("/")
        for commodity in [base, quote]:
            ret_col = f"{commodity.lower()}_daily_return"
            if ret_col in df.columns:
                returns[commodity] = df[ret_col]

    if not returns:
        return

    ret_df = pd.DataFrame(returns).dropna()
    corr   = ret_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Commodity Daily Returns — Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_weather_vs_price(
    pair_name: str, df: pd.DataFrame
) -> None:
    """Three-panel: price, temperature, precipitation for a commodity pair."""
    base, _ = pair_name.split("/")
    price_col = f"{base.lower()}_price"
    if price_col not in df.columns:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{pair_name} — Price vs Weather", fontsize=14, fontweight="bold")

    # Price
    axes[0].plot(df.index, df[price_col], color="#1565C0", lw=1)
    axes[0].set_ylabel(f"{base} Price")
    axes[0].set_title(f"{base} Price")

    # Temperature
    if "base_temp" in df.columns:
        temp_monthly = df["base_temp"].resample("ME").mean()
        axes[1].fill_between(temp_monthly.index, temp_monthly, alpha=0.4, color="#F44336")
        axes[1].plot(temp_monthly.index, temp_monthly, color="#F44336", lw=1)
        axes[1].set_ylabel("Avg Temp (°C)")
        axes[1].set_title("Average Temperature (Production Region Composite)")

    # Precipitation
    if "base_precip" in df.columns:
        precip_monthly = df["base_precip"].resample("ME").sum()
        axes[2].bar(precip_monthly.index, precip_monthly,
                    color="#2196F3", alpha=0.6, width=20)
        axes[2].set_ylabel("Precipitation (mm)")
        axes[2].set_title("Monthly Precipitation")

    plt.tight_layout()
    out = FIG_DIR / f"{pair_name.replace('/', '_').lower()}_weather_vs_price.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out.name}")


def plot_model_comparison(results_table: pd.DataFrame) -> None:
    """Bar chart comparing MAPE across SARIMAX, RF, XGBoost, LSTM."""
    if results_table.empty:
        return

    model_cols = [c for c in ["sarimax_mape", "rf_mape_pct", "xgb_mape_pct", "lstm_mape_pct"]
                  if c in results_table.columns]
    if not model_cols:
        return

    plot_df = results_table.melt(
        id_vars="commodity", value_vars=model_cols,
        var_name="model", value_name="mape_pct"
    )
    model_labels = {
        "sarimax_mape":  "SARIMAX",
        "rf_mape_pct":   "Random Forest",
        "xgb_mape_pct":  "XGBoost",
        "lstm_mape_pct": "LSTM",
    }
    plot_df["model"] = plot_df["model"].map(model_labels)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_df, x="commodity", y="mape_pct", hue="model",
                palette="Set2", ax=ax)
    ax.set_title("Forecast Accuracy Comparison — MAPE % (lower is better)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Commodity")
    ax.set_ylabel("MAPE (%)")
    ax.legend(title="Model")
    plt.tight_layout()
    out = FIG_DIR / "model_comparison_mape.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLOTLY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def build_interactive_dashboard(pairs: dict[str, pd.DataFrame]) -> None:
    """Build and save an interactive HTML dashboard with all key charts."""
    if not pairs:
        return

    pair_list  = list(pairs.keys())
    first_pair = pair_list[0]
    base, quote = first_pair.split("/")
    df          = pairs[first_pair]
    b_col       = f"{base.lower()}_price"
    q_col       = f"{quote.lower()}_price"

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Price Levels", "Rolling Correlation (90d)",
            "Return Scatter", "Spread / Ratio",
            "Temperature vs Price", "Precipitation vs Price",
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]],
    )

    # ── Price levels ──────────────────────────────────────────────────────────
    if b_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[b_col], name=base,
                                 line=dict(color="#1565C0", width=1.2)),
                      row=1, col=1, secondary_y=False)
    if q_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[q_col], name=quote,
                                 line=dict(color="#B71C1C", width=1.2)),
                      row=1, col=1, secondary_y=True)

    # ── Rolling correlation ────────────────────────────────────────────────────
    b_ret = f"{base.lower()}_daily_return"
    q_ret = f"{quote.lower()}_daily_return"
    if b_ret in df.columns and q_ret in df.columns:
        roll_corr = df[b_ret].rolling(90).corr(df[q_ret]).dropna()
        fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr,
                                 name="90d Correlation",
                                 line=dict(color="#4CAF50", width=1.5)),
                      row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # ── Return scatter ─────────────────────────────────────────────────────────
    if b_ret in df.columns and q_ret in df.columns:
        fig.add_trace(go.Scatter(x=df[b_ret], y=df[q_ret], mode="markers",
                                 marker=dict(size=2, color="rgba(33,150,243,0.3)"),
                                 name=f"{base} vs {quote} Returns"),
                      row=2, col=1)

    # ── Spread ────────────────────────────────────────────────────────────────
    if "spread" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["spread"], name="Spread",
                                 line=dict(color="#FF9800", width=1)),
                      row=2, col=2)

    # ── Weather ────────────────────────────────────────────────────────────────
    if "base_temp" in df.columns and b_col in df.columns:
        monthly = df.resample("ME").mean()
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly[b_col],
                                 name=f"{base} Price", line=dict(color="#1565C0")),
                      row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["base_temp"],
                                 name="Temp (°C)", line=dict(color="#F44336", dash="dot")),
                      row=3, col=1, secondary_y=True)

    if "base_precip" in df.columns and b_col in df.columns:
        monthly = df.resample("ME").mean()
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["base_precip"],
                             name="Precipitation", marker_color="rgba(33,150,243,0.4)"),
                      row=3, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly[b_col],
                                 name=f"{base} Price (right)", line=dict(color="#1565C0")),
                      row=3, col=2, secondary_y=True)

    fig.update_layout(
        title_text=f"Commodity Analysis Dashboard — {first_pair}",
        height=1100,
        template="plotly_white",
        hovermode="x unified",
    )

    out = FIG_DIR / "interactive_dashboard.html"
    fig.write_html(str(out))
    logger.info(f"Saved interactive dashboard: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

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

    for pair_name, df in pairs.items():
        plot_pair_prices(pair_name, df)
        plot_returns_distribution(pair_name, df)
        plot_weather_vs_price(pair_name, df)

    plot_correlation_heatmap(pairs)
    build_interactive_dashboard(pairs)

    logger.info("✅ All visualizations complete.")


if __name__ == "__main__":
    main()
