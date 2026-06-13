"""Weekly retraining pipeline for the crop price forecasting project."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.models.arima_model import ARIMAForecaster
from src.models.ensemble_forecaster import EnsemblePriceForecaster
from src.models.lstm_forecaster import LSTMPriceForecaster
from src.models.tabular_forecaster import XGBoostPriceForecaster
from src.mlops.drift import window_drift_report
from src.mlops.prediction_store import PredictionStore
from src.utils.duckdb_client import DuckDBClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_models(series: pd.Series, crop: str, state: str, forecast_days: int = 30) -> dict:
    """Train ARIMA, XGBoost, and LSTM models on one crop-state series."""

    artifacts = {}
    metrics = {}
    arima_test_size = min(30, max(7, len(series) // 10))
    if len(series) <= arima_test_size * 3 + 10:
        arima_test_size = max(7, len(series) // 5)
    xgb_test_size = min(90, max(30, len(series) // 8))
    if len(series) <= xgb_test_size + 30:
        xgb_test_size = max(30, len(series) // 4)

    arima = ARIMAForecaster(crop=crop, state=state, order=(1, 1, 1))
    arima.fit(series)
    artifacts["arima"] = arima.save()
    metrics["arima"] = arima.walk_forward_cv(series, test_size=arima_test_size, n_splits=3)

    xgb = XGBoostPriceForecaster(crop=crop, state=state)
    xgb.fit(series)
    artifacts["xgb"] = xgb.save()
    metrics["xgb"] = xgb.evaluate(series, test_size=xgb_test_size)

    lstm = LSTMPriceForecaster(crop=crop, state=state, epochs=10, max_samples=6000)
    lstm.fit(series)
    artifacts["lstm"] = lstm.save()

    ensemble = EnsemblePriceForecaster(crop=crop, state=state, lstm_epochs=10, lstm_max_samples=6000)
    ensemble.fit(series)
    artifacts["ensemble"] = ensemble.save()
    metrics["ensemble"] = ensemble.evaluation_

    store = PredictionStore()
    for model_name, forecaster in [("arima", arima), ("xgb", xgb), ("lstm", lstm), ("ensemble", ensemble)]:
        forecast_df = forecaster.forecast(steps=forecast_days)
        forecast_df["crop"] = crop
        forecast_df["state"] = state
        forecast_df["model"] = model_name
        store.save_forecast(forecast_df)

    return {"artifacts": artifacts, "metrics": metrics, "weights": ensemble.weights}


def run_pipeline(crop: str, state: str, freq: str = "D") -> dict:
    """End-to-end retrain flow for a single crop-state pair."""

    db = DuckDBClient()
    series = db.crop_series(crop, state, freq=freq)
    try:
        drift = window_drift_report(series)
    except ValueError as exc:
        logger.warning("Drift report skipped: %s", exc)
        drift = {"error": str(exc)}
    training = train_models(series, crop, state)

    return {
        "crop": crop,
        "state": state,
        "series_points": len(series),
        "drift": drift,
        "training": training,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the crop forecasting stack")
    parser.add_argument("--crop", default="Tomato")
    parser.add_argument("--state", default="Tamil Nadu")
    parser.add_argument("--freq", default="D")
    args = parser.parse_args()

    result = run_pipeline(args.crop, args.state, args.freq)
    logger.info("Retrain completed: %s", result)


if __name__ == "__main__":
    main()
