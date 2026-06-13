"""Tabular time-series forecasters used by the Phase 3 ML pipeline.

This module focuses on feature-based forecasting for a single crop-state
series. It is designed for large datasets because it only materializes the
requested series, not the whole raw table.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - dependency issue
    XGBRegressor = None  # type: ignore
    _XGB_IMPORT_ERROR = exc

try:
    import shap
except ImportError as exc:  # pragma: no cover - dependency issue
    shap = None  # type: ignore
    _SHAP_IMPORT_ERROR = exc


ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_LAGS = (1, 7, 14, 28, 56)
DEFAULT_WINDOWS = (7, 14, 30, 90)


def _rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _ensure_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must use a DatetimeIndex")
    return series.sort_index().asfreq("D").ffill().astype(float)


def build_feature_frame(
    series: pd.Series,
    lags: Iterable[int] = DEFAULT_LAGS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    horizon: int = 1,
) -> pd.DataFrame:
    """Convert a time series into a supervised learning frame.
    
    For short series, automatically clips lags and windows to available data.
    """

    series = _ensure_series(series)
    lags = tuple(sorted(set(int(v) for v in lags)))
    windows = tuple(sorted(set(int(v) for v in windows)))
    
    # Clip lags and windows to fit the series length
    # Need at least max_lookback + horizon observations
    max_allowed = len(series) - horizon
    lags = tuple(lag for lag in lags if lag < max_allowed)
    windows = tuple(win for win in windows if win < max_allowed)
    
    # If all were filtered out, use minimal defaults
    if not lags:
        lags = (min(1, max_allowed - 1),) if max_allowed > 1 else (1,)
    if not windows:
        windows = (min(1, max_allowed - 1),) if max_allowed > 1 else (1,)
    
    max_lookback = max(max(lags, default=1), max(windows, default=1))

    rows = []
    values = series.values.astype(float)
    dates = series.index

    for origin in range(max_lookback - 1, len(series) - horizon):
        target_idx = origin + horizon
        history = values[: origin + 1]
        target_date = dates[target_idx]
        row = {
            "date": target_date,
            "target": values[target_idx],
            "year": target_date.year,
            "month": target_date.month,
            "day": target_date.day,
            "dayofweek": target_date.dayofweek,
            "weekofyear": int(target_date.isocalendar().week),
            "quarter": target_date.quarter,
            "is_month_start": int(target_date.is_month_start),
            "is_month_end": int(target_date.is_month_end),
            "is_weekend": int(target_date.dayofweek >= 5),
        }

        for lag in lags:
            row[f"lag_{lag}"] = float(history[-lag])

        for window in windows:
            window_vals = history[-window:]
            row[f"roll_mean_{window}"] = float(np.mean(window_vals))
            row[f"roll_std_{window}"] = float(np.std(window_vals, ddof=0))
            row[f"roll_min_{window}"] = float(np.min(window_vals))
            row[f"roll_max_{window}"] = float(np.max(window_vals))

        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("series is too short for the requested lag/window setup")
    return frame


def train_test_split_timeframe(df: pd.DataFrame, test_size: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test, clipping test_size to fit the data.
    
    For short datasets (< 30 rows), uses 30% for test. For longer datasets,
    respects the requested test_size.
    """
    # For very short datasets, use a percentage-based split
    if len(df) < 30:
        test_size = max(1, len(df) // 3)  # ~33% for test
    else:
        # Ensure test_size doesn't exceed dataset
        test_size = min(test_size, len(df) - 1)
    
    if test_size <= 0 or test_size >= len(df):
        raise ValueError(f"test_size ({test_size}) must be positive and smaller than the dataset ({len(df)})")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def _build_row_from_values(
    values: Sequence[float],
    next_date: pd.Timestamp,
    lags: Iterable[int],
    windows: Iterable[int],
) -> dict:
    """Create the feature row used for the next recursive prediction step.
    
    Automatically filters lags and windows to available data points.
    """

    row = {
        "year": next_date.year,
        "month": next_date.month,
        "day": next_date.day,
        "dayofweek": next_date.dayofweek,
        "weekofyear": int(next_date.isocalendar().week),
        "quarter": next_date.quarter,
        "is_month_start": int(next_date.is_month_start),
        "is_month_end": int(next_date.is_month_end),
        "is_weekend": int(next_date.dayofweek >= 5),
    }

    # Only use lags that fit within the available history
    max_available = len(values)
    valid_lags = [lag for lag in lags if lag < max_available]
    valid_windows = [window for window in windows if window <= max_available]
    
    for lag in valid_lags:
        row[f"lag_{lag}"] = float(values[-lag])

    for window in valid_windows:
        window_vals = values[-window:]
        row[f"roll_mean_{window}"] = float(np.mean(window_vals))
        row[f"roll_std_{window}"] = float(np.std(window_vals, ddof=0))
        row[f"roll_min_{window}"] = float(np.min(window_vals))
        row[f"roll_max_{window}"] = float(np.max(window_vals))

    return row


@dataclass
class XGBoostPriceForecaster:
    """Feature-based price forecaster built on top of XGBoost."""

    crop: str = "unknown"
    state: str = "unknown"
    lags: tuple[int, ...] = field(default_factory=lambda: DEFAULT_LAGS)
    windows: tuple[int, ...] = field(default_factory=lambda: DEFAULT_WINDOWS)
    model_params: dict = field(default_factory=dict)

    model: Optional[XGBRegressor] = field(init=False, default=None)
    feature_columns: list[str] = field(init=False, default_factory=list)
    history_: Optional[pd.Series] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if XGBRegressor is None:  # pragma: no cover - dependency issue
            raise ImportError(
                "xgboost is not installed. Install requirements.txt to use XGBoostPriceForecaster."
            ) from _XGB_IMPORT_ERROR

        defaults = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "tree_method": "hist",
        }
        defaults.update(self.model_params)
        self.model_params = defaults

    def fit(self, series: pd.Series) -> "XGBoostPriceForecaster":
        series = _ensure_series(series)
        frame = build_feature_frame(series, self.lags, self.windows)
        self.feature_columns = [c for c in frame.columns if c not in {"date", "target"}]

        self.model = XGBRegressor(**self.model_params)
        self.model.fit(frame[self.feature_columns], frame["target"])
        self.history_ = series
        return self

    def evaluate(self, series: pd.Series, test_size: int = 90) -> dict:
        series = _ensure_series(series)
        frame = build_feature_frame(series, self.lags, self.windows)
        train_df, test_df = train_test_split_timeframe(frame, test_size=test_size)

        model = XGBRegressor(**self.model_params)
        model.fit(train_df[self.feature_columns or [c for c in train_df.columns if c not in {"date", "target"}]], train_df["target"])
        feature_cols = self.feature_columns or [c for c in train_df.columns if c not in {"date", "target"}]
        preds = model.predict(test_df[feature_cols])

        return {
            "rmse": round(_rmse(test_df["target"], preds), 2),
            "mae": round(float(mean_absolute_error(test_df["target"], preds)), 2),
            "mape": round(float(np.mean(np.abs((test_df["target"] - preds) / test_df["target"])) * 100), 2),
            "n_test": int(len(test_df)),
        }

    def forecast(self, steps: int = 30, history: Optional[pd.Series] = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")

        if history is None:
            if self.history_ is None:
                raise RuntimeError("No history available. Pass a series or call fit() first.")
            history = self.history_

        history = _ensure_series(history)
        values = history.values.astype(float).tolist()
        last_date = history.index[-1]
        rows = []

        for step in range(1, steps + 1):
            next_date = last_date + pd.Timedelta(days=step)
            row = _build_row_from_values(values, next_date, self.lags, self.windows)
            x = pd.DataFrame([row], columns=self.feature_columns)
            pred = float(self.model.predict(x)[0])
            values.append(pred)
            rows.append({"date": next_date, "forecast": round(pred, 2), "model": "xgboost"})

        return pd.DataFrame(rows)

    def explain_next_step(
        self,
        history: Optional[pd.Series] = None,
        top_n: int = 10,
    ) -> dict:
        """Return SHAP-based contributions for the next forecast step.

        The result is intentionally small and serialisable so it can be shown in
        the API or dashboard without a separate plotting dependency.
        """

        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        if shap is None:  # pragma: no cover - dependency issue
            raise ImportError(
                "shap is not installed. Install requirements.txt to use explainability."
            ) from _SHAP_IMPORT_ERROR

        if history is None:
            if self.history_ is None:
                raise RuntimeError("No history available. Pass a series or call fit() first.")
            history = self.history_

        history = _ensure_series(history)
        values = history.values.astype(float).tolist()
        next_date = history.index[-1] + pd.Timedelta(days=1)
        row = _build_row_from_values(values, next_date, self.lags, self.windows)
        x = pd.DataFrame([row], columns=self.feature_columns)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values).reshape(-1)

        prediction = float(self.model.predict(x)[0])
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        contributions = []
        for feature, value, shap_value in zip(self.feature_columns, x.iloc[0].tolist(), shap_values):
            contributions.append(
                {
                    "feature": feature,
                    "value": float(value),
                    "shap_value": float(shap_value),
                }
            )

        top_positive = sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)[:top_n]
        return {
            "forecast_date": str(next_date.date()),
            "prediction": round(prediction, 2),
            "base_value": round(base_value, 4),
            "top_contributions": top_positive,
        }

    def save(self, path: Optional[str] = None) -> str:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        path = path or str(ARTIFACT_DIR / f"xgb_{self.crop.lower().replace(' ', '_')}_{self.state.lower().replace(' ', '_')}.pkl")
        payload = {
            "crop": self.crop,
            "state": self.state,
            "lags": self.lags,
            "windows": self.windows,
            "model_params": self.model_params,
            "feature_columns": self.feature_columns,
            "model": self.model,
            "history": self.history_,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: str) -> "XGBoostPriceForecaster":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            crop=payload["crop"],
            state=payload["state"],
            lags=tuple(payload["lags"]),
            windows=tuple(payload["windows"]),
            model_params=payload["model_params"],
        )
        obj.model = payload["model"]
        obj.feature_columns = list(payload["feature_columns"])
        obj.history_ = payload.get("history")
        return obj
