"""Utilities for locating and materializing forecasting model artifacts."""

from __future__ import annotations

from pathlib import Path

from src.models.arima_model import ARIMAForecaster
from src.models.ensemble_forecaster import EnsemblePriceForecaster
from src.models.lstm_forecaster import LSTMPriceForecaster
from src.models.tabular_forecaster import XGBoostPriceForecaster
from src.utils.duckdb_client import DuckDBClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def artifact_path(model: str, crop: str, state: str) -> Path:
    """Return the canonical artifact path for a model/crop/state combination."""
    tag = f"{crop.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}"
    if model == "arima":
        return MODELS_DIR / f"arima_{tag}.pkl"
    if model == "xgb":
        return MODELS_DIR / f"xgb_{tag}.pkl"
    if model == "lstm":
        return MODELS_DIR / f"lstm_{tag}.pt"
    if model == "ensemble":
        return MODELS_DIR / f"ensemble_{tag}.pkl"
    raise ValueError(f"Unknown model type: {model}")


def ensure_model_artifact(model_name: str, crop: str, state: str) -> tuple[object, Path]:
    """Load an existing model or train and save one if it is missing."""
    artifact = artifact_path(model_name, crop, state)
    if artifact.exists():
        if model_name == "arima":
            return ARIMAForecaster.load(str(artifact)), artifact
        if model_name == "xgb":
            return XGBoostPriceForecaster.load(str(artifact)), artifact
        if model_name == "ensemble":
            return EnsemblePriceForecaster.load(str(artifact)), artifact
        return LSTMPriceForecaster.load(str(artifact)), artifact

    db = DuckDBClient()
    series = db.crop_series(crop, state, freq="D")

    if model_name == "arima":
        model = ARIMAForecaster(crop=crop, state=state, order=(1, 1, 1))
    elif model_name == "xgb":
        model = XGBoostPriceForecaster(crop=crop, state=state)
    elif model_name == "ensemble":
        model = EnsemblePriceForecaster(crop=crop, state=state)
    else:
        model = LSTMPriceForecaster(crop=crop, state=state)

    model.fit(series)
    model.save(str(artifact))
    return model, artifact
