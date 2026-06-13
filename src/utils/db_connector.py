"""Lightweight local database connector for optional preprocessing writes.

The original project documentation refers to a DBConnector helper, but the
implementation was missing. This module provides a small DuckDB-backed
connector that is good enough for local development and for writing cleaned
dataframes into a persistent analytics store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class DBConnector:
    """Minimal DuckDB-based connector with DataFrame write support."""

    def __init__(self, db_path: str = "data/crop_prices.duckdb", threads: int = 4):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute(f"SET threads={threads}")

    def query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def scalar(self, sql: str):
        return self.conn.execute(sql).fetchone()[0]

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> None:
        """Persist a DataFrame into a DuckDB table.

        Parameters
        ----------
        df:
            The DataFrame to write.
        table_name:
            Destination table.
        if_exists:
            "append" or "replace".
        """

        if if_exists not in {"append", "replace"}:
            raise ValueError("if_exists must be either 'append' or 'replace'")

        view_name = "__df_write_view__"
        self.conn.register(view_name, df)

        if if_exists == "replace":
            self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
        else:
            table_exists = (
                self.conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_name = ?
                    """,
                    [table_name],
                ).fetchone()[0]
                > 0
            )
            if not table_exists:
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {view_name} WHERE 1=0")
            self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM {view_name}")

        self.conn.unregister(view_name)

