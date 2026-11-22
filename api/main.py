"""FastAPI prediction entry point for the crop forecasting platform."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.arima_model import ARIMAForecaster
from src.models.ensemble_forecaster import EnsemblePriceForecaster
from src.models.lstm_forecaster import LSTMPriceForecaster
from src.models.tabular_forecaster import XGBoostPriceForecaster
from src.mlops.advisory import generate_crop_advisory
from src.mlops.drift import window_drift_report
from src.mlops.model_artifacts import ensure_model_artifact
from src.mlops.prediction_store import PredictionStore
from src.utils.duckdb_client import DuckDBClient


app = FastAPI(title="Crop Price Forecast API", version="0.1.0")


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
    raise ValueError(f"Unknown model type: {model}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast")
@app.get("/predict")
def forecast(
    crop: str = Query(..., description="Commodity name"),
    state: str = Query(..., description="State name"),
    steps: int = Query(30, ge=1, le=365),
    model: str = Query("arima", pattern="^(arima|xgb|lstm|ensemble)$"),
) -> dict:
    artifact = _artifact_path(model, crop, state)
    forecaster, artifact = ensure_model_artifact(model, crop, state)

    if model == "arima":
        forecast_df = forecaster.forecast(steps=steps)
    elif model == "xgb":
        forecast_df = forecaster.forecast(steps=steps)
    elif model == "ensemble":
        forecast_df = forecaster.forecast(steps=steps)
    else:
        forecast_df = forecaster.forecast(steps=steps)

    history = None
    if hasattr(forecaster, "history_"):
        history = getattr(forecaster, "history_")
    if history is None:
        db = DuckDBClient()
        history = db.crop_series(crop, state, freq="D")

    advisory = generate_crop_advisory(history=history, forecast_df=forecast_df, explanation=None)

    return {
        "crop": crop,
        "state": state,
        "model": model,
        "steps": steps,
        "forecast": forecast_df.to_dict(orient="records"),
        "advisory": advisory,
    }


@app.get("/explain")
def explain(
    crop: str = Query(..., description="Commodity name"),
    state: str = Query(..., description="State name"),
    model: str = Query("xgb", pattern="^(xgb|ensemble)$"),
    top_n: int = Query(10, ge=1, le=20),
) -> dict:
    """Return SHAP-style feature contributions for the next XGBoost or ensemble forecast."""

    artifact = _artifact_path(model, crop, state)
    if not artifact.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact not found: {artifact}. Train the requested model first.",
        )

    if model == "ensemble":
        forecaster = EnsemblePriceForecaster.load(str(artifact))
    else:
        forecaster = XGBoostPriceForecaster.load(str(artifact))
    explanation = forecaster.explain_next_step(top_n=top_n)
    return {
        "crop": crop,
        "state": state,
        "model": model,
        "explanation": explanation,
    }


@app.get("/series")
def series(crop: str = Query(...), state: str = Query(...), freq: str = Query("D")) -> dict:
    db = DuckDBClient()
    s = db.crop_series(crop, state, freq=freq)
    return {
        "crop": crop,
        "state": state,
        "points": len(s),
        "start": str(s.index[0].date()),
        "end": str(s.index[-1].date()),
    }


@app.get("/advisory")
def advisory(
    crop: str = Query(..., description="Commodity name"),
    state: str = Query(..., description="State name"),
    model: str = Query("ensemble", pattern="^(arima|xgb|lstm|ensemble)$"),
    steps: int = Query(30, ge=1, le=365),
    top_n: int = Query(8, ge=1, le=20),
) -> dict:
    """Return a crop advisory derived from the selected model forecast."""

    artifact = _artifact_path(model, crop, state)
    if not artifact.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact not found: {artifact}. Train the requested model first.",
        )

    if model == "arima":
        forecaster = ARIMAForecaster.load(str(artifact))
    elif model == "xgb":
        forecaster = XGBoostPriceForecaster.load(str(artifact))
    elif model == "ensemble":
        forecaster = EnsemblePriceForecaster.load(str(artifact))
    else:
        forecaster = LSTMPriceForecaster.load(str(artifact))

    forecast_df = forecaster.forecast(steps=steps)
    history = getattr(forecaster, "history_", None)
    if history is None:
        db = DuckDBClient()
        history = db.crop_series(crop, state, freq="D")

    explanation = None
    if model in {"xgb", "ensemble"} and hasattr(forecaster, "explain_next_step"):
        explanation = forecaster.explain_next_step(top_n=top_n)

    return {
        "crop": crop,
        "state": state,
        "model": model,
        "steps": steps,
        "advisory": generate_crop_advisory(history=history, forecast_df=forecast_df, explanation=explanation),
    }


@app.get("/history")
def history(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Return recent stored forecast rows from the local prediction log."""

    store = PredictionStore()
    recent = store.recent_forecasts(limit=limit)
    return {
        "limit": limit,
        "records": recent.to_dict(orient="records"),
    }


@app.get("/monitoring")
def monitoring(
    crop: str = Query(..., description="Commodity name"),
    state: str = Query(..., description="State name"),
    freq: str = Query("D"),
    history_limit: int = Query(50, ge=1, le=500),
) -> dict:
    """Return forecast history and a lightweight drift summary."""

    series = None
    try:
        forecaster, _ = ensure_model_artifact("xgb", crop, state)
        if hasattr(forecaster, "history_") and getattr(forecaster, "history_") is not None:
            series = getattr(forecaster, "history_")
    except Exception:
        pass

    if series is None or len(series) < 455:
        db = DuckDBClient()
        series = db.crop_series(crop, state, freq=freq)
    store = PredictionStore()
    recent = store.recent_forecasts(limit=history_limit)
    try:
        drift = window_drift_report(series)
    except ValueError as exc:
        drift = {"error": str(exc)}
    return {
        "crop": crop,
        "state": state,
        "drift": drift,
        "recent_forecasts": recent.to_dict(orient="records"),
    }
