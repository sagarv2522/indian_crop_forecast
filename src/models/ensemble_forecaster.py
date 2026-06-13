"""Weighted ensemble forecaster for crop price prediction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.arima_model import ARIMAForecaster
from src.models.lstm_forecaster import LSTMPriceForecaster
from src.models.tabular_forecaster import XGBoostPriceForecaster


ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must use a DatetimeIndex")
    return series.sort_index().asfreq("D").ffill().astype(float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(y_true == 0, 1e-6, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}


@dataclass
class EnsemblePriceForecaster:
    """Weighted ensemble of ARIMA, XGBoost, and LSTM forecasts."""

    crop: str = "unknown"
    state: str = "unknown"
    validation_days: int = 90
    lstm_epochs: int = 10
    lstm_max_samples: int = 6000
    xgb_params: dict = field(default_factory=dict)

    weights: dict = field(init=False, default_factory=dict)
    evaluation_: dict = field(init=False, default_factory=dict)
    history_: Optional[pd.Series] = field(init=False, default=None)
    models_: dict = field(init=False, default_factory=dict)

    def _build_component_models(self) -> dict:
        return {
            "arima": ARIMAForecaster(crop=self.crop, state=self.state, order=(1, 1, 1)),
            "xgb": XGBoostPriceForecaster(crop=self.crop, state=self.state, model_params=self.xgb_params),
            "lstm": LSTMPriceForecaster(
                crop=self.crop,
                state=self.state,
                epochs=self.lstm_epochs,
                max_samples=self.lstm_max_samples,
            ),
        }

    def fit(self, series: pd.Series) -> "EnsemblePriceForecaster":
        series = _ensure_series(series)
        validation_days = min(self.validation_days, max(14, len(series) // 5))
        if len(series) <= validation_days + 30:
            raise ValueError("series is too short for ensemble validation")
        self.validation_days = validation_days

        train = series.iloc[:-validation_days]
        valid = series.iloc[-validation_days :]

        component_models = self._build_component_models()
        scores = {}

        for name, model in component_models.items():
            model.fit(train)
            forecast_df = model.forecast(steps=self.validation_days)
            preds = forecast_df["forecast"].to_numpy(dtype=float)
            actual = valid.to_numpy(dtype=float)
            scores[name] = _metrics(actual, preds)

        inverse = {}
        for name, metrics in scores.items():
            inverse[name] = 1.0 / max(metrics["mape"], 1e-6)
        total = sum(inverse.values()) or 1.0
        self.weights = {name: round(weight / total, 4) for name, weight in inverse.items()}
        self.evaluation_ = scores

        # Refit on the full series so the ensemble is production-ready.
        self.models_ = self._build_component_models()
        for model in self.models_.values():
            model.fit(series)
        self.history_ = series
        return self

    def forecast(self, steps: int = 30, history: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("Model not fitted yet.")

        forecast_frames = {}
        for name, model in self.models_.items():
            if name == "arima":
                forecast_frames[name] = model.forecast(steps=steps)
            else:
                forecast_frames[name] = model.forecast(steps=steps, history=history)

        base = forecast_frames["arima"][["date"]].copy()
        weighted = np.zeros(len(base), dtype=float)
        for name, frame in forecast_frames.items():
            weighted += frame["forecast"].to_numpy(dtype=float) * self.weights.get(name, 0.0)
            base[f"{name}_forecast"] = frame["forecast"].to_numpy(dtype=float)

        base["forecast"] = np.round(weighted, 2)
        base["model"] = "ensemble"
        return base[["date", "forecast", "model", "arima_forecast", "xgb_forecast", "lstm_forecast"]]

    def explain_next_step(self, top_n: int = 10) -> dict:
        if not self.models_:
            raise RuntimeError("Model not fitted yet.")
        xgb = self.models_["xgb"]
        explanation = xgb.explain_next_step(top_n=top_n)
        explanation["ensemble_weights"] = self.weights
        explanation["component_metrics"] = self.evaluation_
        return explanation

    def save(self, path: Optional[str] = None) -> str:
        if not self.models_:
            raise RuntimeError("Model not fitted yet.")
        path = path or str(
            ARTIFACT_DIR / f"ensemble_{self.crop.lower().replace(' ', '_')}_{self.state.lower().replace(' ', '_')}.pkl"
        )
        payload = {
            "crop": self.crop,
            "state": self.state,
            "validation_days": self.validation_days,
            "lstm_epochs": self.lstm_epochs,
            "lstm_max_samples": self.lstm_max_samples,
            "xgb_params": self.xgb_params,
            "weights": self.weights,
            "evaluation": self.evaluation_,
            "history": self.history_,
            "models": self.models_,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: str) -> "EnsemblePriceForecaster":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            crop=payload["crop"],
            state=payload["state"],
            validation_days=payload["validation_days"],
            lstm_epochs=payload["lstm_epochs"],
            lstm_max_samples=payload["lstm_max_samples"],
            xgb_params=payload["xgb_params"],
        )
        obj.weights = payload["weights"]
        obj.evaluation_ = payload["evaluation"]
        obj.history_ = payload.get("history")
        obj.models_ = payload["models"]
        return obj
