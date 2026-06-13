"""Streamlit dashboard for crop price forecasting."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.prediction_store import PredictionStore
from src.mlops.advisory import generate_crop_advisory
from src.mlops.drift import window_drift_report
from src.mlops.model_artifacts import ensure_model_artifact
from src.utils.duckdb_client import DuckDBClient


st.set_page_config(page_title="🌾 Smart Crop Price Forecast Platform", layout="wide")


def _artifact_path(model: str, crop: str, state: str) -> Path:
    tag = f"{crop.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}"
    if model == "arima":
        return Path("models") / f"arima_{tag}.pkl"
    if model == "xgb":
        return Path("models") / f"xgb_{tag}.pkl"
    if model == "lstm":
        return Path("models") / f"lstm_{tag}.pt"
    if model == "ensemble":
        return Path("models") / f"ensemble_{tag}.pkl"
    raise ValueError(model)


@st.cache_data(show_spinner=False)
def load_series(crop: str, state: str) -> pd.Series:
    db = DuckDBClient()
    return db.crop_series(crop, state, freq="D")


def load_model(model_name: str, crop: str, state: str):
    return ensure_model_artifact(model_name, crop, state)


st.title("Indian Crop Price Forecasting")
st.caption("DuckDB-backed dashboard for ARIMA, XGBoost, LSTM, and ensemble forecasts.")

db = DuckDBClient()
col1, col2, col3 = st.columns(3)
with col1:
    crops = sorted(db.query("SELECT DISTINCT commodity AS crop FROM prices ORDER BY crop")["crop"].tolist())
    crop = st.selectbox("Crop Selection", crops, index=crops.index("Tomato") if "Tomato" in crops else 0)
with col2:
    states = sorted(db.query("SELECT DISTINCT state FROM prices ORDER BY state")["state"].tolist())
    state = st.selectbox("State Selection", states, index=states.index("Tamil Nadu") if "Tamil Nadu" in states else 0)
with col3:
    model_name = st.selectbox("Model Selection", ["arima", "xgb", "lstm", "ensemble"], index=0, help="Hover for model guidance")

steps = st.slider("Forecast horizon (days)", 7, 180, 30, 1)

if st.button("Generate forecast"):
    with st.spinner("Loading series and model..."):
        try:
            series = load_series(crop, state)

        except ValueError:
            st.markdown(
                """
                <div style="
                    padding:20px;
                    border-radius:15px;
                    background-color:#fff8e1;
                    border-left:8px solid #ff9800;
                    margin-top:20px;
                ">
                    <h3>⚠️ Data Not Available</h3>
                    <p>
                        We could not find enough historical data for the selected
                        Crop and State combination.
                    </p>
                    <p>
                        Please select another State or Crop to continue forecasting.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()
        model, artifact = load_model(model_name, crop, state)
        forecast_df = model.forecast(steps=steps)
        st.session_state["latest_forecast"] = forecast_df
        st.session_state["latest_crop"] = crop
        st.session_state["latest_state"] = state
        st.session_state["latest_model"] = model_name

    fig = go.Figure()
    history = series.iloc[-1000:]
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], name="Forecast", line=dict(width=2, dash="dash")))
    fig.update_layout(
        title=f"{crop} @ {state} - {model_name.upper()} forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast_df, use_container_width=True)

    explanation = None
    if model_name in {"xgb", "ensemble"} and hasattr(model, "explain_next_step"):
        with st.expander("Explain next forecast with SHAP"):
            explanation = model.explain_next_step(top_n=8)
            st.write(
                {
                    "forecast_date": explanation["forecast_date"],
                    "prediction": explanation["prediction"],
                    "base_value": explanation["base_value"],
                }
            )
            if "ensemble_weights" in explanation:
                st.write(
                    {
                        "ensemble_weights": explanation["ensemble_weights"],
                        "component_metrics": explanation["component_metrics"],
                    }
                )
            st.dataframe(pd.DataFrame(explanation["top_contributions"]), use_container_width=True)

    advisory = generate_crop_advisory(history=series, forecast_df=forecast_df, explanation=explanation)
    st.subheader("Crop Advisory")
    st.write(
        {
            "recommendation": advisory["recommendation"],
            "confidence": advisory["confidence"],
            "metrics": advisory["metrics"],
        }
    )
    for item in advisory["rationale"]:
        st.write(f"- {item}")

latest_forecast = st.session_state.get("latest_forecast")
if latest_forecast is not None and st.button("Save forecast log"):
    store = PredictionStore()
    store.save_forecast(
        latest_forecast.assign(
            crop=st.session_state.get("latest_crop", crop),
            state=st.session_state.get("latest_state", state),
            model=st.session_state.get("latest_model", model_name),
        )
    )
    st.success("Forecast saved to the local prediction log.")


st.divider()
st.subheader("Prediction History and Drift")

history_col1, history_col2 = st.columns([2, 1])
with history_col1:
    limit = st.slider("Recent forecast rows", 5, 100, 20, 5)
    store = PredictionStore()
    recent = store.recent_forecasts(limit=limit)
    st.dataframe(recent, use_container_width=True)

with history_col2:
    try:
        drift = window_drift_report(load_series(crop, state))
        st.write(
            {
                "baseline_mean": drift["baseline_mean"],
                "recent_mean": drift["recent_mean"],
                "mean_shift_pct": drift["mean_shift_pct"],
                "median_shift_pct": drift["median_shift_pct"],
                "psi": drift["psi"],
            }
        )
    except ValueError as exc:
        st.info(f"Drift summary unavailable yet: {exc}")