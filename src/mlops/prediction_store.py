"""Persist forecast outputs for drift analysis and dashboard use."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd


class PredictionStore:
    """Simple local forecast log backed by DuckDB."""

    def __init__(self, db_path: str = "data/prediction_logs.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_log (
                crop VARCHAR,
                state VARCHAR,
                model VARCHAR,
                forecast_date DATE,
                predicted_price DOUBLE,
                lower_80 DOUBLE,
                upper_80 DOUBLE,
                lower_95 DOUBLE,
                upper_95 DOUBLE,
                created_at TIMESTAMP
            )
            """
        )

    def save_forecast(self, forecast_df: pd.DataFrame) -> None:
        """Save a forecast dataframe into the local log table."""

        df = forecast_df.copy()
        if "forecast_date" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "forecast_date"})

        df["created_at"] = datetime.utcnow()
        keep_cols = [
            "crop", "state", "model", "forecast_date", "forecast",
            "lower_80", "upper_80", "lower_95", "upper_95", "created_at",
        ]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = None

        insert_df = df[keep_cols].rename(columns={"forecast": "predicted_price"})
        self.conn.register("forecast_df", insert_df)
        self.conn.execute("INSERT INTO prediction_log SELECT * FROM forecast_df")
        self.conn.unregister("forecast_df")

    def recent_forecasts(self, limit: int = 100) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT *
            FROM prediction_log
            ORDER BY created_at DESC, forecast_date DESC
            LIMIT ?
            """,
            [limit],
        ).df()
