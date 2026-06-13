# Crop Price Forecasting & Market Intelligence Platform

Primary setup/run guide: [PROJECT_RUN_AND_SETUP.txt](E:/Program%20Files/vs%20code/Crop_prediction/PROJECT_RUN_AND_SETUP.txt)

> End-to-end ML + DL + MLOps + Statistics system for forecasting Indian mandi crop prices.
> Built on publicly available government data — fully reproducible, zero cost.

---

## Architecture

```
data.gov.in ──► download_data.py ──► preprocessor.py ──► PostgreSQL
                                                               │
agmarknet.gov.in ──► scraper.py (weekly CI/CD) ──────────────►│
                                                               │
                                              ┌────────────────┘
                                              ▼
                                   Feature Engineering
                                              │
                          ┌───────────────────┼──────────────┐
                          ▼                   ▼              ▼
                       ARIMA              XGBoost          LSTM
                          └───────────────────┼──────────────┘
                                         Ensemble
                                              │
                                    FastAPI /predict
                                              │
                                      Streamlit App
```

---

## Quickstart — Phase 1 Setup

Use [PROJECT_RUN_AND_SETUP.txt](E:/Program%20Files/vs%20code/Crop_prediction/PROJECT_RUN_AND_SETUP.txt) as the main step-by-step reference. The section below is a shorter copy inside the README for convenience.

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/crop-price-forecaster.git
cd crop-price-forecaster

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — for local dev, the SQLite default requires no changes
```

### 3. Download data

```bash
# Download all states (recommended — takes ~5 min)
python -m src.data.download_data

# Or download a specific crop + state for quick testing
python -m src.data.download_data --crop "Tomato" --state "Tamil Nadu"
```

### 4. Preprocess

```bash
python -m src.data.preprocessor --input data/raw/mandi_prices_all.csv
# Clean CSV saved to: data/processed/clean_prices.csv
```

### 5. Run EDA

```bash
python notebooks/01_eda.py
# All charts saved to: notebooks/eda_outputs/
```

### 6. Train Forecast Models

```bash
# Statistical baseline
python notebooks/02_statistical_models.py --crop "Tomato" --state "Tamil Nadu"

# XGBoost feature model
python notebooks/03_ml_models.py --crop "Tomato" --state "Tamil Nadu"

# LSTM sequence model
python notebooks/04_lstm_model.py --crop "Tomato" --state "Tamil Nadu"

# Ensemble model
python run_pipeline.py --train --crop "Tomato" --state "Tamil Nadu"
```

### 7. Run the API and dashboard

```bash
# FastAPI
uvicorn api.main:app --reload

# Streamlit
streamlit run app/streamlit_app.py
```

### 8. Weekly retraining

```bash
python run_pipeline.py --train --crop "Tomato" --state "Tamil Nadu"
```

---

## Project Structure

```
crop-price-forecaster/
├── .github/workflows/          ← CI/CD: scraper + retrain
├── data/
│   ├── raw/                    ← downloaded CSVs
│   ├── processed/              ← clean_prices.csv
│   └── features/               ← engineered feature sets
├── database/
│   └── schema.sql              ← PostgreSQL DDL
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_statistical_models.py
│   ├── 03_ml_models.py
│   └── 04_lstm_model.py
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   └── preprocessor.py
│   ├── models/                 ← Phase 3
│   ├── mlops/                  ← Phase 4
│   └── utils/
│       └── db_connector.py
├── app/                        ← Phase 5: Streamlit
├── api/                        ← Phase 5: FastAPI
├── .env.example
├── requirements.txt
├── PROJECT_RUN_AND_SETUP.txt    ← main setup + run guide
├── QUICK_START.txt              ← minimal command order
├── SETUP_CHECKLIST.txt          ← environment prerequisites
├── RUN_FULL_PROJECT.txt         ← step-by-step runbook
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | Pandas, SQLAlchemy, PostgreSQL / SQLite |
| Statistics | statsmodels, pmdarima, scipy |
| ML | XGBoost, scikit-learn, Optuna, SHAP |
| Deep Learning | PyTorch (LSTM) |
| MLOps | MLflow, Evidently, GitHub Actions |
| Web App | Streamlit, FastAPI, Plotly |
| Deployment | Streamlit Cloud, Supabase, Docker |

---

## Data Sources

| Source | URL | License |
|---|---|---|
| Historical mandi prices | data.gov.in | Open Government Data (OGD) India |
| Weekly live prices | agmarknet.gov.in | Public government data |

---

## Phases

- [x] **Phase 1** — Data collection & EDA *(current)*
- [ ] **Phase 2** — Statistical modelling (ARIMA, SARIMA, STL)
- [ ] **Phase 3** — ML + DL models (XGBoost, LSTM, Ensemble)
- [ ] **Phase 4** — MLOps pipeline (MLflow, drift detection, CI/CD retrain)
- [ ] **Phase 5** — Web app (Streamlit + FastAPI)
- [ ] **Phase 6** — Deployment (Streamlit Cloud + Supabase)

## Automation

- `.github/workflows/weekly_retrain.yml` runs weekly and can be triggered manually.
- The workflow downloads the latest government data, converts it to Parquet, and retrains the model stack.
- The FastAPI `/explain` endpoint returns SHAP-style feature contributions for the XGBoost or ensemble forecast.
- The FastAPI `/advisory` endpoint and Streamlit dashboard both provide a crop advisory on top of the forecast.
- The FastAPI `/history` and `/monitoring` endpoints expose recent stored predictions and a lightweight drift summary.
- Run `python -m src.mlops.project_status` to quickly check which core artifacts and data outputs are present.
