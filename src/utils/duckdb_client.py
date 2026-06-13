"""
src/utils/duckdb_client.py
───────────────────────────
Single DuckDB connection that reads ALL Parquet partitions as one
virtual table called `prices`. No full dataset ever loads into RAM —
DuckDB scans only the columns and partitions your query touches.

Usage:
    from src.utils.duckdb_client import DuckDBClient

    db = DuckDBClient()

    # Total record count across all 10 years
    print(db.scalar("SELECT COUNT(*) FROM prices"))

    # Pull a crop-market time series as a DataFrame
    df = db.crop_series("Tomato", "Koyambedu")

    # Raw SQL → DataFrame
    df = db.query(\"\"\"
        SELECT state, commodity, MEDIAN(modal_price) AS median_price
        FROM prices
        WHERE year BETWEEN 2020 AND 2024
        GROUP BY state, commodity
        ORDER BY median_price DESC
        LIMIT 20
    \"\"\")
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PARQUET_DIR = Path("data/parquet")
LOCAL_GLOB  = str(PARQUET_DIR / "**" / "*.parquet")

# ── Azure Blob config (read from .env / environment) ──────────────────────
DATA_SOURCE      = os.getenv("DATA_SOURCE", "auto")          # "azure" | "local" | "auto"
AZURE_BASE_URL   = os.getenv("AZURE_BLOB_BASE_URL", "https://cropspricesstorage.blob.core.windows.net/crops-data").rstrip("/")
AZURE_SAS_TOKEN  = os.getenv("AZURE_SAS_TOKEN", r"sp=rl&st=2026-06-12T16:40:11Z&se=2027-01-01T00:55:11Z&spr=https&sv=2026-02-06&sr=c&sig=i5w62HmXa6RzO1sZXmtMZZQ41hjy3rjqa3Sxur%2FossQ%3D").lstrip("?")
AZURE_YEAR_START = int(os.getenv("AZURE_YEAR_START", "2016"))
AZURE_YEAR_END   = int(os.getenv("AZURE_YEAR_END", "2026"))


def _norm(s: str) -> str:
    """Normalise whitespace and title-case a commodity / market name."""
    return re.sub(r"\s+", " ", s.strip().title())


class DuckDBClient:
    """
    Thin wrapper around a DuckDB in-process connection.

    The `prices` view covers all Parquet partitions under
    data/parquet/year=YYYY/ via a glob pattern.
    Hive-style partitioning is enabled so WHERE year=2022 triggers
    partition pruning — only that year's file is read.
    """

    def __init__(self, db_path: str = ":memory:", threads: int = 4, source: Optional[str] = None):
        """
        Args:
            db_path : ":memory:" (default) or path to a .duckdb file
                      for a persistent connection.
            threads : number of CPU threads DuckDB may use.
            source  : "azure" | "local" | None (uses DATA_SOURCE env, default "auto")
                      "auto" picks Azure if AZURE_BLOB_BASE_URL + AZURE_SAS_TOKEN
                      are set, else falls back to local Parquet.
        """
        self.conn = duckdb.connect(db_path)
        self.conn.execute(f"SET threads={threads}")

        chosen = source or DATA_SOURCE
        use_azure = (
            chosen == "azure"
            or (chosen == "auto" and AZURE_BASE_URL and AZURE_SAS_TOKEN)
        )

        if use_azure:
            self._init_azure()
        else:
            self._init_local()

    def _init_azure(self) -> None:
        """
        Register the `prices` view from Parquet files in Azure Blob Storage,
        read directly over HTTPS using a SAS token via DuckDB's httpfs.

        Expects one file per year: data_2016.parquet ... data_2026.parquet
        under AZURE_BLOB_BASE_URL, each already containing a `year` column
        (written by src/data/json_to_parquet.py).
        """
        if not (AZURE_BASE_URL and AZURE_SAS_TOKEN):
            raise RuntimeError(
                "Azure source selected but AZURE_BLOB_BASE_URL / AZURE_SAS_TOKEN "
                "are not set. Check your .env file."
            )

        self.conn.execute("INSTALL httpfs;")
        self.conn.execute("LOAD httpfs;")

        urls = [
            f"{AZURE_BASE_URL}/data_{year}.parquet?{AZURE_SAS_TOKEN}"
            for year in range(AZURE_YEAR_START, AZURE_YEAR_END + 1)
        ]
        url_list_sql = ", ".join(f"'{u}'" for u in urls)

        self.conn.execute(f"""
            CREATE OR REPLACE VIEW prices AS
            SELECT * FROM read_parquet([{url_list_sql}])
        """)

        logger.info(
            r"DuckDB ready — Azure Blob source, %d yearly file(s) (%d–%d)",
            len(urls), AZURE_YEAR_START, AZURE_YEAR_END,
        )

    def _init_local(self) -> None:
        """Register the `prices` view from local data/parquet/year=YYYY/*.parquet."""
        if not PARQUET_DIR.exists():
            raise FileNotFoundError(
                f"Parquet directory not found: {PARQUET_DIR.resolve()}\n"
                "Run  python -m src.data.json_to_parquet  first, "
                "or set DATA_SOURCE=azure in .env to use Azure Blob."
            )

        parts = list(PARQUET_DIR.glob("**/*.parquet"))
        if not parts:
            raise FileNotFoundError(
                f"No Parquet files found in {PARQUET_DIR.resolve()}\n"
                "Run  python -m src.data.json_to_parquet  first, "
                "or set DATA_SOURCE=azure in .env to use Azure Blob."
            )

        self.conn.execute(f"""
            CREATE OR REPLACE VIEW prices AS
            SELECT *
            FROM read_parquet(
                '{LOCAL_GLOB}',
                hive_partitioning = true
            )
        """)

        logger.info(
            r"DuckDB ready — local source, %d partition file(s) registered as view 'prices'",
            len(parts),
        )

    # ── Core query methods ─────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute any SQL and return a pandas DataFrame."""
        return self.conn.execute(sql).df()

    def scalar(self, sql: str):
        """Execute SQL that returns a single value."""
        return self.conn.execute(sql).fetchone()[0]

    # ── Catalogue helpers ──────────────────────────────────────────────────

    def overview(self) -> dict:
        """High-level stats across the full dataset."""
        row = self.conn.execute("""
            SELECT
                COUNT(*)                            AS total_rows,
                COUNT(DISTINCT commodity)           AS unique_crops,
                COUNT(DISTINCT market)              AS unique_markets,
                COUNT(DISTINCT state)               AS unique_states,
                MIN(arrival_date)                   AS date_from,
                MAX(arrival_date)                   AS date_to,
                ROUND(AVG(modal_price), 2)          AS avg_modal_price,
                ROUND(MEDIAN(modal_price), 2)       AS median_modal_price
            FROM prices
        """).fetchone()
        keys = [
            "total_rows", "unique_crops", "unique_markets",
            "unique_states", "date_from", "date_to",
            "avg_modal_price", "median_modal_price",
        ]
        return dict(zip(keys, row))

    def list_crops(self, min_records: int = 100) -> pd.DataFrame:
        """Return all commodities with at least min_records records."""
        return self.query(f"""
            SELECT
                commodity,
                COUNT(*)                      AS records,
                COUNT(DISTINCT market)        AS markets,
                COUNT(DISTINCT state)         AS states,
                ROUND(MEDIAN(modal_price), 2) AS median_price
            FROM prices
            GROUP BY commodity
            HAVING COUNT(*) >= {min_records}
            ORDER BY records DESC
        """)

    def list_state(self, commodity: str = None) -> pd.DataFrame:
        """List states for a commodity ordered by record count."""
        where = f"WHERE commodity = '{commodity}'" if commodity else ""

        return self.query(f"""
            SELECT
                state,
                COUNT(*)               AS records,
                MIN(arrival_date)      AS first_date,
                MAX(arrival_date)      AS last_date
            FROM prices
            {where}
            GROUP BY state
            ORDER BY records DESC
        """)

    # ── Time-series extraction ─────────────────────────────────────────────

    def crop_series(
        self,
        commodity: str,
        state: str,
        freq: str = "W",
    ) -> pd.Series:
        """
        Extract a modal_price time series for one commodity-market pair.

        Args:
            commodity : exact name as stored (title-cased)
            market    : exact name as stored  (use list_markets() to check)
            freq      : resample frequency — 'D' daily, 'W' weekly, 'MS' monthly

        Returns:
            pd.Series with DatetimeIndex, name='modal_price', NaN-forward-filled.
        """
        nc = _norm(commodity)
        ns = _norm(state)

        df = self.query(f"""
            SELECT
                arrival_date,
                MEDIAN(modal_price) AS modal_price
            FROM prices
            WHERE
                regexp_replace(commodity, '\\s+', ' ') = '{nc}'
                AND regexp_replace(state, '\\s+', ' ') = '{ns}'
            GROUP BY arrival_date
            ORDER BY arrival_date
        """)

        if df.empty:
            raise ValueError(
                f"No data found for commodity='{nc}', state='{ns}'.\n"
                f"Check  db.list_state('{nc}')  for valid state names."
            )

        series = (
            df.set_index(pd.to_datetime(df["arrival_date"]))["modal_price"]
            .asfreq("D")
            .ffill()
            .resample(freq)
            .median()
            .dropna()
        )
        series.name = "modal_price"
        logger.info(
            "Series loaded: %s @ %s — %d %s-frequency points (%s → %s)",
            nc, ns, len(series), freq,
            series.index[0].date(), series.index[-1].date(),
        )
        return series

    def state_summary(self, year: int = None) -> pd.DataFrame:
        """Median price per state. Optionally filter by year."""
        where = f"WHERE year = {year}" if year else ""
        return self.query(f"""
            SELECT
                state,
                COUNT(DISTINCT commodity)     AS crops,
                COUNT(DISTINCT market)        AS markets,
                COUNT(*)                      AS records,
                ROUND(MEDIAN(modal_price), 2) AS median_price,
                ROUND(STDDEV(modal_price), 2) AS price_stddev
            FROM prices
            {where}
            GROUP BY state
            ORDER BY records DESC
        """)

    def seasonal_avg(self, commodity: str) -> pd.DataFrame:
        """Monthly average price for one commodity across all years."""
        nc = _norm(commodity)
        return self.query(f"""
            SELECT
                month,
                ROUND(AVG(modal_price),    2) AS avg_price,
                ROUND(MEDIAN(modal_price), 2) AS median_price,
                ROUND(STDDEV(modal_price), 2) AS price_stddev,
                COUNT(*)                      AS records
            FROM prices
            WHERE regexp_replace(commodity, '\\s+', ' ') = '{nc}'
            GROUP BY month
            ORDER BY month
        """)

    def yearly_trend(self, commodity: str = None) -> pd.DataFrame:
        """Year-on-year median price trend."""
        if commodity:
            nc = _norm(commodity)
            where = f"WHERE regexp_replace(commodity, '\\s+', ' ') = '{nc}'"
        else:
            where = ""
        return self.query(f"""
            SELECT
                year,
                ROUND(MEDIAN(modal_price), 2) AS median_price,
                ROUND(AVG(modal_price),    2) AS avg_price,
                COUNT(*)                      AS records
            FROM prices
            {where}
            GROUP BY year
            ORDER BY year
        """)

    def top_volatile_markets(
        self, commodity: str, top_n: int = 10
    ) -> pd.DataFrame:
        """Markets with highest price volatility for a given commodity."""
        nc = _norm(commodity)
        return self.query(f"""
            SELECT
                market,
                state,
                ROUND(STDDEV(modal_price), 2)                    AS volatility,
                ROUND(MEDIAN(modal_price), 2)                    AS median_price,
                ROUND(STDDEV(modal_price) / AVG(modal_price), 3) AS cv,
                COUNT(*)                                         AS records
            FROM prices
            WHERE regexp_replace(commodity, '\\s+', ' ') = '{nc}'
            GROUP BY market, state
            HAVING COUNT(*) >= 52
            ORDER BY volatility DESC
            LIMIT {top_n}
        """)

    def price_correlation(self, commodity: str) -> pd.DataFrame:
        """Monthly median price correlation between top states."""
        nc = _norm(commodity)
        return self.query(f"""
            SELECT
                year,
                month,
                state,
                ROUND(MEDIAN(modal_price), 2) AS median_price
            FROM prices
            WHERE regexp_replace(commodity, '\\s+', ' ') = '{nc}'
            GROUP BY year, month, state
            ORDER BY year, month, state
        """)

    def close(self) -> None:
        self.conn.close()
        logger.info("DuckDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()