"""Data-driven crop advisory generation for forecast outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CropAdvisory:
    recommendation: str
    confidence: str
    rationale: list[str]
    metrics: dict


def _safe_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        raise ValueError("history series cannot be empty")
    return s


def generate_crop_advisory(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    explanation: Optional[dict] = None,
) -> dict:
    """Create a simple market advisory from history + forecast signals."""

    history = _safe_series(history)
    if forecast_df.empty:
        raise ValueError("forecast_df cannot be empty")

    forecast = forecast_df["forecast"].astype(float)
    last_price = float(history.iloc[-1])
    first_price = float(forecast.iloc[0])
    last_forecast = float(forecast.iloc[-1])

    change_pct = ((last_forecast - last_price) / max(last_price, 1e-6)) * 100.0
    forecast_volatility = float(forecast.std(ddof=0) / max(forecast.mean(), 1e-6) * 100.0)
    recent_volatility = float(history.tail(min(30, len(history))).std(ddof=0) / max(history.tail(min(30, len(history))).mean(), 1e-6) * 100.0)
    slope = float(np.polyfit(np.arange(len(forecast)), forecast.values, 1)[0]) if len(forecast) > 1 else 0.0

    rationale: list[str] = []
    if change_pct >= 5:
        recommendation = "Hold stock or sell in phases"
        confidence = "High" if slope > 0 else "Medium"
        rationale.append(f"Forecast price is expected to rise by about {change_pct:.1f}% over the horizon.")
        rationale.append("A phased sale may capture higher prices while reducing timing risk.")
    elif change_pct <= -5:
        recommendation = "Sell sooner"
        confidence = "High" if slope < 0 else "Medium"
        rationale.append(f"Forecast price is expected to fall by about {abs(change_pct):.1f}% over the horizon.")
        rationale.append("Earlier selling may protect margins before prices soften.")
    else:
        recommendation = "Hold / stagger sales"
        confidence = "Medium"
        rationale.append("Forecast is relatively flat, so sharp timing bets may not add much value.")
        rationale.append("Staggering sales can help balance risk and price capture.")

    if forecast_volatility >= 8 or recent_volatility >= 8:
        rationale.append("Price volatility is elevated, so avoid a single large market decision.")
        confidence = "Medium" if confidence == "High" else confidence

    if explanation and explanation.get("top_contributions"):
        top_feature = explanation["top_contributions"][0]["feature"]
        rationale.append(f"The forecast is most influenced by {top_feature}, which indicates recent market momentum matters.")

    metrics = {
        "last_price": round(last_price, 2),
        "first_forecast": round(first_price, 2),
        "last_forecast": round(last_forecast, 2),
        "change_pct": round(change_pct, 2),
        "forecast_volatility_pct": round(forecast_volatility, 2),
        "recent_volatility_pct": round(recent_volatility, 2),
        "slope": round(slope, 4),
    }

    advisory = CropAdvisory(
        recommendation=recommendation,
        confidence=confidence,
        rationale=rationale,
        metrics=metrics,
    )
    return advisory.__dict__

