<div align="center">

# 🌾 Indian Crop Price Forecasting Platform

**End-to-end ML + Deep Learning + MLOps system for forecasting Indian mandi crop prices**

Built on 10 years of publicly available government data — fully reproducible, zero cost.

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://inidan-crops-state-forecast0.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-Analytics-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)](https://duckdb.org)
[![Azure](https://img.shields.io/badge/Azure%20Blob-Storage-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)

</div>

---

## Overview

This platform forecasts crop prices across Indian agricultural markets (mandis) using a multi-model ensemble pipeline. Historical price data spanning 2016–2026 is stored as Parquet files on Azure Blob Storage and queried in-process via DuckDB — no database server required. The Streamlit dashboard lets users select any crop-state pair, choose a forecasting model, and generate forecasts with actionable market advisories.

---

## Architecture

```
data.gov.in ──► download_data.py ──► json_to_parquet.py ──► Azure Blob Storage
                                                                      │
                                                         DuckDB (httpfs over HTTPS)
                                                                      │
                                              ┌───────────────────────┤
                                              ▼                       ▼
                                     Feature Engineering         EDA / Stats
                                              │
                          ┌───────────────────┼──────────────────┐
                          ▼                   ▼                  ▼
                       ARIMA              XGBoost              LSTM
                          └───────────────────┼──────────────────┘
                                      Weighted Ensemble
                                              │
                                   Crop Advisory Engine
                                              │
                                      Streamlit App
```

The data layer is **serverless** — DuckDB reads Parquet files directly from Azure Blob over HTTPS using its built-in `httpfs` extension. No local copy of the data is needed to run the app.

---

## Models

| Model | Type | Description |
|---|---|---|
| **ARIMA** | Statistical | Auto-selected order via AIC; handles trend and seasonality with differencing |
| **XGBoost** | Gradient Boosting | Lag features, rolling statistics, calendar encodings; SHAP explanations |
| **LSTM** | Deep Learning (PyTorch) | Sequence model with sliding window; captures non-linear temporal patterns |
| **Ensemble** | Weighted combination | Performance-weighted blend of all three; weights derived from validation MAE |

---

## Features

**Forecasting**
- 7 to 180-day price forecasts for any crop-state combination in the dataset
- Historical price chart with forecast overlay (Plotly, interactive)
- Downloadable forecast table

**Market Advisory**
- Data-driven buy/sell/hold recommendations from forecast signals
- Confidence scoring based on price trend slope and volatility
- SHAP feature attribution for XGBoost and Ensemble models

**Drift Monitoring**
- Population Stability Index (PSI) between baseline and recent price windows
- Mean and median shift percentages for early warning on price regime changes

**Prediction History**
- Local prediction store logs all generated forecasts
- Configurable recent-forecast viewer built into the dashboard

---

## Data

| Source | Coverage | Format |
|---|---|---|
| [data.gov.in Mandi Prices](https://data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070) | 2016 – 2026, all India | Parquet on Azure Blob |

**Dataset stats (as of 2026):**

- 11 yearly Parquet files (`data_2016.parquet` → `data_2026.parquet`)
- Fields: `state`, `district`, `market`, `commodity`, `variety`, `grade`, `arrival_date`, `min_price`, `max_price`, `modal_price`
- DuckDB queries with `WHERE year = YYYY` trigger partition pruning — only the relevant file is read

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data Storage** | Azure Blob Storage, Parquet |
| **Query Engine** | DuckDB (in-process, httpfs for Azure) |
| **Statistics** | statsmodels, pmdarima, scipy |
| **ML** | XGBoost, scikit-learn, Optuna, SHAP |
| **Deep Learning** | PyTorch (LSTM) |
| **MLOps** | MLflow, Evidently, GitHub Actions |
| **Web App** | Streamlit, Plotly |
| **Data Pipeline** | Pandas, PyArrow |
| **Deployment** | Streamlit Cloud, Python 3.11 |

---

## Local Setup

### Prerequisites

- Python 3.11
- Azure Blob SAS token (or use `DATA_SOURCE=local` with local Parquet files)

### Install

```bash
git clone https://github.com/sagarv2522/indian_crop_forecast.git
cd indian_crop_forecast

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
# Set DATA_SOURCE=azure and add your AZURE_BLOB_BASE_URL + AZURE_SAS_TOKEN
# Or set DATA_SOURCE=local and run json_to_parquet.py first
```

| Variable | Description |
|---|---|
| `DATA_SOURCE` | `azure` \| `local` \| `auto` (default: auto) |
| `AZURE_BLOB_BASE_URL` | Base URL to the Azure Blob container |
| `AZURE_SAS_TOKEN` | SAS token with read+list permissions |
| `AZURE_YEAR_START` | First year to load (default: 2016) |
| `AZURE_YEAR_END` | Last year to load (default: 2026) |

### Download and Prepare Data (local mode)

```bash
# Download historical data from data.gov.in (last 2 years by default)
python -m src.data.download_data --crop "Tomato" --state "Tamil Nadu"

# Or use monthly chunks for large crop-state combos (avoids API timeouts)
python -m src.data.download_data --crop "Tomato" --state "Tamil Nadu" \
    --from-date 2022-01-01 --chunked

# Convert raw JSON/CSV to Parquet
python -m src.data.json_to_parquet
```

### Train Models

```bash
# Statistical baseline (ARIMA/SARIMA)
python notebooks/02_statistical_models.py --crop "Tomato" --state "Tamil Nadu"

# XGBoost feature model
python notebooks/03_ml_models.py --crop "Tomato" --state "Tamil Nadu"

# LSTM sequence model
python notebooks/04_lstm_model.py --crop "Tomato" --state "Tamil Nadu"

# Full ensemble pipeline
python run_pipeline.py --train --crop "Tomato" --state "Tamil Nadu"
```

### Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## DuckDB Query Examples

The `DuckDBClient` exposes the full dataset as a SQL view called `prices`:

```python
from src.utils.duckdb_client import DuckDBClient

db = DuckDBClient()

# Total records across all years
db.scalar("SELECT COUNT(*) FROM prices")

# Top volatile markets for Tomato
db.top_volatile_markets("Tomato", top_n=10)

# Monthly price seasonality
db.seasonal_avg("Onion")

# Raw time series for one crop-state pair
series = db.crop_series("Tomato", "Tamil Nadu", freq="W")
```

DuckDB scans only the columns and year-partitions your query touches — queries on a single year read one file regardless of total dataset size.

---

## Automation

A GitHub Actions workflow runs weekly to download the latest government data, convert it to Parquet, push to Azure Blob, and retrain the model stack. It can also be triggered manually from the Actions tab.

---

## Project Status

- [x] **Phase 1** — Data collection, EDA, Azure Blob + DuckDB pipeline
- [x] **Phase 2** — Statistical modelling (ARIMA, SARIMA, STL decomposition)
- [x] **Phase 3** — ML + DL models (XGBoost, LSTM, weighted Ensemble)
- [x] **Phase 4** — MLOps (drift monitoring, prediction store, advisory engine)
- [x] **Phase 5** — Streamlit dashboard (live at the link above)

---

## Data License

Price data sourced from [data.gov.in](https://data.gov.in) under the **Open Government Data (OGD) Platform India** license — free for non-commercial and research use.
