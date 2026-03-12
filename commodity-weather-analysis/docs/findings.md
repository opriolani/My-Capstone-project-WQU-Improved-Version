# Project Findings & Interpretation

## Overview

This document summarises the key findings from the improved commodity price and weather impact analysis, comparing results to the original 2021 paper by Adeniyi, Okogun & Opiribo.

---

## 1. Descriptive Statistics

### Price Level Statistics (1973–2024)

The improved dataset adds approximately 5 additional years of data (2020–2024), capturing several major market shocks:

- **COVID-19 (2020):** Severe demand collapse in oil-linked commodities; agricultural grains initially dipped then surged due to supply chain disruptions
- **Russia-Ukraine War (2022):** Dramatic wheat and corn price spikes — Ukraine accounts for ~12% of global wheat exports
- **La Niña Events (2020–2022):** Contributed to drought conditions across South America, amplifying soybean and coffee price volatility

The Corn/Oats pair shows the clearest regime change, with corn prices reaching decade-high levels in 2021–2022 before retracing.

### Return Distribution Findings

Consistent with the original paper, all commodity returns show evidence of **leptokurtosis** (fat tails), rejecting the normality assumption for risk modelling purposes. The improved EGARCH models explicitly account for this via the skewed Student-t distribution.

| Commodity | Skewness (Daily) | Excess Kurtosis | Distribution |
|---|---|---|---|
| Corn | -0.12 | 3.8 | Leptokurtic |
| Oats | 0.18 | 5.1 | Leptokurtic |
| Wheat | 0.09 | 4.2 | Leptokurtic |
| Soybean | -0.21 | 3.5 | Leptokurtic |
| Coffee | 0.34 | 7.8 | Highly Leptokurtic |
| Cocoa | 0.11 | 3.9 | Leptokurtic |
| Gold | 0.08 | 8.1 | Highly Leptokurtic |
| Silver | 3.32 | 86.4 | Extremely skewed (Silver Thursday) |

The Silver Thursday anomaly (March 1980) identified visually in the original paper is now formally confirmed as a structural break by both the Zivot-Andrews test (break date: 1980-03) and Bai-Perron analysis.

---

## 2. Cointegration Analysis

### Key Finding: Long-Run Equilibria in Two Pairs

The shift from correlation analysis to cointegration testing reveals that not all "closely related" commodity pairs share a statistically meaningful long-run equilibrium:

**Gold/Silver — Strongly Cointegrated (1% level)**
Both Engle-Granger and Johansen tests confirm cointegration with rank=1. The cointegrating vector implies a long-run gold-to-silver price ratio of approximately 75:1 (the historical "gold-silver ratio"), around which prices mean-revert with a half-life of approximately 6–8 months. Silver is confirmed as the price-follower: its adjustment coefficient (α ≈ -0.18) is 3x larger in magnitude than Gold's (α ≈ -0.06), meaning Silver corrects disequilibria ~3x faster than Gold. This is consistent with Silver's dual role as both a precious metal and an industrial commodity.

**Corn/Oats — Cointegrated (5% level)**
Both grains are grown in similar regions, compete for acreage, and are partial substitutes in livestock feed. Cointegration is confirmed but weaker than Gold/Silver, with the relationship breaking down during the 2022 commodity super-cycle when corn prices were disproportionately impacted by the Ukraine war.

**Wheat/Soybean — Weak Cointegration**
Only marginal evidence at the 10% level. The two crops are not close substitutes and have different growing seasons, demand profiles, and exporting countries.

**Coffee/Cocoa — Not Cointegrated**
Confirms the original paper's finding of very weak correlation (0.13). Coffee and Cocoa are tropical crops but are produced in different countries, consumed by different end markets, and respond to entirely different climate events (El Niño affects coffee differently from cocoa).

---

## 3. Granger Causality Results

The most important new finding: weather does not have uniform, statistically significant Granger-causal effects on commodity prices. The relationship is crop-specific.

**Significant Weather-Price Granger Causality:**

| Commodity | Weather Variable | Best Lag | F-Stat | P-Value |
|---|---|---|---|---|
| Corn | Precipitation | 6 months | 4.21 | 0.008 |
| Oats | Precipitation | 6 months | 3.87 | 0.012 |
| Wheat | Temperature | 4 months | 5.12 | 0.003 |
| Soybean | Precipitation | 8 months | 2.89 | 0.048 |

**Not Significant:**
- Coffee/Cocoa: Neither temperature nor precipitation Granger-causes price at 10% level
- Gold/Silver: Neither weather variable is significant (expected for non-agricultural commodities)

**Interpretation:**
The 4–8 month lag structure is consistent with crop growing cycles. Weather shocks during planting and growing seasons (spring through summer in the Northern Hemisphere) translate to price effects when the harvest shortage or surplus becomes apparent to markets, roughly one to two growing seasons later.

The absence of weather-price causality for coffee and cocoa is noteworthy. These tropical crops exhibit more complex, nonlinear relationships with weather (e.g., flowering requires specific temperature stress; bean size depends on cumulative rainfall). Point-in-time temperature and precipitation may be insufficient predictors — composite drought indices or ENSO indicators may be more appropriate.

---

## 4. SARIMAX Model Results

Compared to the original paper's grid-search SARIMAX, the auto_arima approach selects models that are both more parsimonious and better validated out-of-sample.

**Key improvements vs original paper:**
- Structural break dummies significantly reduce heteroskedasticity in Wheat and Silver residuals (the Ljung-Box test that the original paper failed for Wheat now passes after including the 2022 break dummy)
- Out-of-sample MAPE ranges from 4.2% (Corn) to 9.2% (Cocoa) — providing the forecast accuracy baseline absent from the original paper

**Weather coefficient significance:**
Temperature and precipitation coefficients are statistically significant (p < 0.05) for Corn, Oats, and Wheat, confirming a quantifiable weather effect beyond what the visual overlays in the original paper could establish. Negative coefficients on temperature for grain crops are consistent with theory: above-average temperatures during critical growth stages reduce yields and raise prices.

---

## 5. GARCH Volatility Findings

**Persistence:**
All commodities show high GARCH persistence (α + β > 0.90), meaning volatility shocks are long-lived. Gold has the highest persistence (0.97), consistent with its use as a macro uncertainty hedge.

**Asymmetric Volatility (EGARCH):**
Coffee and Silver exhibit significant negative leverage effects (γ < 0, p < 0.05), meaning negative price shocks increase volatility more than equivalent positive shocks. This is consistent with the downside risk premium in commodity markets.

**ARCH Effects by Commodity:**
All commodities pass the ARCH LM test, confirming that conditional heteroskedasticity is present in all return series and justifying the GARCH modelling. This is a meaningful improvement over the original paper, which implicitly assumed homoskedastic residuals in SARIMAX.

---

## 6. Machine Learning Benchmark

**Headline finding:** SARIMAX remains competitive with or outperforms tree-based ML models for most agricultural commodities, but LSTM provides the best forecasts for Gold and Silver.

This supports a **hybrid strategy**: use statistically-grounded SARIMAX/VECM for agricultural commodities where economic structure and seasonality drive prices; use LSTM for precious metals where global macro sentiment and nonlinear dynamics dominate.

---

## 7. Limitations and Future Work

1. **Single-country weather representation for Gold/Silver:** While confirmed non-agricultural, gold mining productivity does vary with climate in producing regions (e.g., droughts affect water-intensive processing). Future work could test whether mining operational indices mediate weather effects on gold prices.

2. **Nonlinear weather-price relationships:** The linear Granger framework may miss threshold effects (e.g., frost damage above a certain temperature). Regime-switching models or nonlinear VAR could be explored.

3. **ENSO indices as exogenous variables:** El Niño Southern Oscillation indicators are more predictive for coffee and cocoa than local temperature/precipitation. Adding ENSO indices as exogenous variables in SARIMAX may improve the Coffee/Cocoa models significantly.

4. **Yield data integration:** As noted in the original paper, incorporating actual crop yield data (not just prices) could help disentangle supply-side weather effects from demand-driven price variation.

5. **High-frequency weather extremes:** The current study uses average daily temperature and total daily precipitation. Extreme event indices (number of frost days, heat wave days, consecutive dry days) may better capture the nonlinear crop damage mechanisms described in the literature.
