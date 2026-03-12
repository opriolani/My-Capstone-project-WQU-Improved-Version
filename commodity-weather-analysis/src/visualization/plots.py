"""
plots.py
--------
Interactive Plotly visualisations for the commodity-weather analysis.

Improvement over original paper: replaces all static matplotlib figures
with interactive Plotly charts that allow zooming, hovering, and
dynamic time-window selection.

Charts available:
    - plot_price_pairs()          → Dual-axis price pair charts
    - plot_returns_distribution() → Return histogram + Q-Q plot
    - plot_correlation_heatmap()  → Interactive correlation matrix
    - plot_weather_vs_price()     → Weather overlay on commodity prices
    - plot_conditional_volatility() → GARCH conditional volatility
    - plot_forecast()             → Price forecast with confidence intervals
    - plot_spread()               → Cointegrated pair spread / ratio
    - plot_structural_breaks()    → Price with structural break markers
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional

COMMODITY_PAIRS = [
    ("Corn",    "Oats"),
    ("Wheat",   "Soybean"),
    ("Coffee",  "Cocoa"),
    ("Gold",    "Silver"),
]

COLOURS = {
    "Corn":    "#F4A460",
    "Oats":    "#DEB887",
    "Wheat":   "#DAA520",
    "Soybean": "#6B8E23",
    "Coffee":  "#6F4E37",
    "Cocoa":   "#3B1A08",
    "Gold":    "#FFD700",
    "Silver":  "#C0C0C0",
}


# ---------------------------------------------------------------------------
# 1. Price pair charts
# ---------------------------------------------------------------------------

def plot_price_pairs(prices: pd.DataFrame, save_html: bool = False) -> go.Figure:
    """Plot all four commodity pairs with dual Y-axes.

    Replicates and improves original paper figures 4.4a–4.4d.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{a} vs {b}" for a, b in COMMODITY_PAIRS],
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]],
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, ((a, b), (row, col)) in enumerate(zip(COMMODITY_PAIRS, positions)):
        if a not in prices.columns or b not in prices.columns:
            continue

        fig.add_trace(
            go.Scatter(x=prices.index, y=prices[a], name=a,
                       line=dict(color=COLOURS.get(a, "#1f77b4"), width=1.5),
                       hovertemplate=f"{a}: %{{y:.2f}}<br>%{{x}}"),
            row=row, col=col, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=prices.index, y=prices[b], name=b,
                       line=dict(color=COLOURS.get(b, "#ff7f0e"), width=1.5,
                                 dash="dash"),
                       hovertemplate=f"{b}: %{{y:.2f}}<br>%{{x}}"),
            row=row, col=col, secondary_y=True,
        )

    fig.update_layout(
        title="Commodity Price Pairs — Daily Historical Prices",
        height=700, width=1200,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.1),
        hovermode="x unified",
    )

    if save_html:
        fig.write_html("results/figures/price_pairs.html")
    return fig


# ---------------------------------------------------------------------------
# 2. Returns distribution + Q-Q plots
# ---------------------------------------------------------------------------

def plot_returns_distribution(
    returns: pd.DataFrame,
    commodity: str,
) -> go.Figure:
    """Histogram of returns with normal overlay and Q-Q plot side by side.

    Replicates and improves original paper Figure 4.1 (Q-Q plots).
    """
    r = returns[commodity].dropna()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Return Distribution", "Q-Q Plot"])

    # Histogram
    fig.add_trace(
        go.Histogram(x=r, nbinsx=80, name="Returns",
                     histnorm="probability density",
                     marker_color=COLOURS.get(commodity, "#636EFA"),
                     opacity=0.75),
        row=1, col=1,
    )
    # Normal overlay
    x_range = np.linspace(r.min(), r.max(), 200)
    normal_pdf = stats.norm.pdf(x_range, r.mean(), r.std())
    fig.add_trace(
        go.Scatter(x=x_range, y=normal_pdf, mode="lines", name="Normal",
                   line=dict(color="red", width=2)),
        row=1, col=1,
    )

    # Q-Q plot
    qq = stats.probplot(r, dist="norm")
    theoretical_q = qq[0][0]
    sample_q      = qq[0][1]
    fit_line_x    = np.array([theoretical_q.min(), theoretical_q.max()])
    fit_line_y    = qq[1][1] + qq[1][0] * fit_line_x

    fig.add_trace(
        go.Scatter(x=theoretical_q, y=sample_q, mode="markers", name="Quantiles",
                   marker=dict(color=COLOURS.get(commodity, "#636EFA"),
                               size=4, opacity=0.6)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=fit_line_x, y=fit_line_y, mode="lines", name="Normal line",
                   line=dict(color="red", width=2)),
        row=1, col=2,
    )

    skew = round(r.skew(), 3)
    kurt = round(r.kurtosis(), 3)
    fig.update_layout(
        title=f"{commodity} Return Distribution  |  Skew={skew}  Kurt={kurt}",
        height=450, width=900, template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    freq_label: str = "Daily",
) -> go.Figure:
    """Interactive correlation heatmap for all commodity returns.

    Replicates and improves original paper Figures 4.2a–4.2c.
    """
    corr = returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Correlation Matrix — {freq_label} Returns",
        height=600, width=700,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Weather vs price overlay
# ---------------------------------------------------------------------------

def plot_weather_vs_price(
    prices: pd.DataFrame,
    weather: dict,
    pair: tuple,
) -> go.Figure:
    """Plot commodity pair prices with temperature and precipitation overlaid.

    Replicates and improves original paper Figures 4.4a–4.4d.
    """
    a, b   = pair
    w_a    = weather.get(a, pd.DataFrame())
    w_b    = weather.get(b, pd.DataFrame())

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.45, 0.275, 0.275],
        subplot_titles=[f"{a} & {b} Prices",
                        f"Temperature — {a} & {b} Production Regions",
                        f"Precipitation — {a} & {b} Production Regions"],
        shared_xaxes=True,
    )

    # Panel 1: Prices
    for name in [a, b]:
        if name in prices.columns:
            fig.add_trace(
                go.Scatter(x=prices.index, y=prices[name], name=name,
                           line=dict(color=COLOURS.get(name), width=1.5)),
                row=1, col=1,
            )

    # Panel 2: Temperature
    for name, wdf in [(a, w_a), (b, w_b)]:
        if not wdf.empty and "TAVG" in wdf.columns:
            fig.add_trace(
                go.Scatter(x=wdf.index, y=wdf["TAVG"],
                           name=f"Temp ({name})",
                           line=dict(color=COLOURS.get(name), width=1, dash="dot"),
                           opacity=0.8),
                row=2, col=1,
            )

    # Panel 3: Precipitation
    for name, wdf in [(a, w_a), (b, w_b)]:
        if not wdf.empty and "PRCP" in wdf.columns:
            fig.add_trace(
                go.Scatter(x=wdf.index, y=wdf["PRCP"],
                           name=f"Precip ({name})",
                           line=dict(color=COLOURS.get(name), width=1, dash="dashdot"),
                           fill="tozeroy", opacity=0.4),
                row=3, col=1,
            )

    fig.update_layout(
        title=f"Weather Impact on {a} / {b}",
        height=800, width=1100,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Temp (°C)",   row=2, col=1)
    fig.update_yaxes(title_text="Precip (mm)", row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# 5. GARCH conditional volatility
# ---------------------------------------------------------------------------

def plot_conditional_volatility(
    returns: pd.Series,
    cond_vol: pd.Series,
    commodity: str,
) -> go.Figure:
    """Plot return series alongside GARCH-estimated conditional volatility."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{commodity} Daily Returns",
                                        "GARCH Conditional Volatility (Annualised)"])

    fig.add_trace(
        go.Scatter(x=returns.index, y=returns * 100, mode="lines",
                   name="Returns (%)",
                   line=dict(color=COLOURS.get(commodity, "#636EFA"), width=0.8)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=cond_vol.index, y=cond_vol * 100, mode="lines",
                   name="Cond. Vol. (%)",
                   line=dict(color="red", width=1.5),
                   fill="tozeroy", fillcolor="rgba(255,0,0,0.1)"),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{commodity} — Returns & GARCH Conditional Volatility",
        height=550, width=1100,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Forecast with confidence intervals
# ---------------------------------------------------------------------------

def plot_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    commodity: str,
    n_history: int = 60,
) -> go.Figure:
    """Plot last n_history periods of actual prices + forecast with CI band."""
    hist = history.iloc[-n_history:]

    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist,
        mode="lines", name="Historical",
        line=dict(color=COLOURS.get(commodity, "#636EFA"), width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="orange", width=2, dash="dash"),
    ))

    # CI band
    if "upper_ci" in forecast_df.columns and "lower_ci" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df.index.to_series(),
                         forecast_df.index.to_series()[::-1]]),
            y=pd.concat([forecast_df["upper_ci"], forecast_df["lower_ci"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

    # Vertical line at forecast start
    split_date = forecast_df.index[0]
    fig.add_vline(x=split_date, line_dash="dot", line_color="grey",
                  annotation_text="Forecast start")

    fig.update_layout(
        title=f"{commodity} — Price Forecast with 95% Confidence Interval",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=500, width=1100,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Spread / ratio for cointegrated pair
# ---------------------------------------------------------------------------

def plot_spread(
    prices: pd.DataFrame,
    pair: tuple,
    hedge_ratio: float = 1.0,
) -> go.Figure:
    """Plot the spread (A - β×B) for a cointegrated pair with mean-reversion bands."""
    a, b = pair
    spread = prices[a] - hedge_ratio * prices[b]
    mean   = spread.mean()
    std    = spread.std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread, mode="lines",
                             name="Spread", line=dict(color="steelblue", width=1)))
    fig.add_hline(y=mean,        line_dash="solid", line_color="red",   annotation_text="Mean")
    fig.add_hline(y=mean + std,  line_dash="dash",  line_color="orange", annotation_text="+1σ")
    fig.add_hline(y=mean - std,  line_dash="dash",  line_color="orange", annotation_text="-1σ")
    fig.add_hline(y=mean + 2*std, line_dash="dot",  line_color="green",  annotation_text="+2σ")
    fig.add_hline(y=mean - 2*std, line_dash="dot",  line_color="green",  annotation_text="-2σ")

    fig.update_layout(
        title=f"Spread: {a} − {hedge_ratio:.2f}×{b}  |  Mean-Reversion Bands",
        xaxis_title="Date", yaxis_title="Spread",
        height=450, width=1100,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Structural break visualisation
# ---------------------------------------------------------------------------

def plot_structural_breaks(
    prices: pd.Series,
    break_dates: list,
) -> go.Figure:
    """Plot price series with vertical lines at identified structural break dates."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices,
        mode="lines", name=prices.name,
        line=dict(color=COLOURS.get(prices.name, "#636EFA"), width=1.5),
    ))

    colours_breaks = ["red", "orange", "purple", "green", "darkred"]
    for i, bd in enumerate(break_dates):
        fig.add_vline(
            x=pd.Timestamp(bd),
            line_dash="dash",
            line_color=colours_breaks[i % len(colours_breaks)],
            annotation_text=f"Break {bd[:7]}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"{prices.name} Price — Structural Break Dates",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=450, width=1100,
        template="plotly_white",
        hovermode="x",
    )
    return fig


if __name__ == "__main__":
    # Quick demo with random data
    import plotly.io as pio
    pio.renderers.default = "browser"

    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=1000, freq="B")
    gold   = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5), index=dates, name="Gold")
    silver = pd.Series(50  + np.cumsum(np.random.randn(1000) * 0.3), index=dates, name="Silver")
    prices = pd.DataFrame({"Gold": gold, "Silver": silver})

    fig = plot_spread(prices, pair=("Gold", "Silver"), hedge_ratio=2.0)
    fig.show()
