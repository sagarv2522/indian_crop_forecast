"""Utility to find and select states with sufficient historical coverage."""

import logging
from src.utils.duckdb_client import DuckDBClient

logger = logging.getLogger(__name__)


def find_best_state(commodity: str, min_records: int = 1000, top_n: int = 1) -> str:
    """Find state(s) with the most historical data for a commodity.
    
    Args:
        commodity: Commodity name (e.g., 'Tomato')
        min_records: Minimum records required (~3 years of daily data)
        top_n: Return top N states by record count (default 1 = best state)
    
    Returns:
        Best state name with most historical records
    """
    db = DuckDBClient()
    
    df = db.query(f"""
        SELECT 
            market,
            state,
            COUNT(*) as total_records,
            MIN(arrival_date) as date_from,
            MAX(arrival_date) as date_to,
            ROUND((MAX(arrival_date) - MIN(arrival_date))::float / 365.25, 1) as years_of_data
        FROM prices
        WHERE regexp_replace(commodity, '\\s+', ' ') = '{commodity.title()}'
        GROUP BY market, state
        HAVING COUNT(*) >= {min_records}
        ORDER BY total_records DESC
        LIMIT {top_n}
    """)
    
    if df.empty:
        raise ValueError(
            f"No states found for '{commodity}' with ≥{min_records} records. "
            f"Try a lower min_records value or check commodity name."
        )
    
    best = df.iloc[0]
    logger.info(
        f"Selected state: {best['state']} — "
        f"{int(best['total_records'])} records, "
        f"{best['years_of_data']} years "
        f"({best['date_from']} → {best['date_to']})"
    )
    
    return str(best['state'])


def suggest_state(commodity: str, limit: int = 5) -> None:
    """Print top states with good coverage for a commodity."""
    db = DuckDBClient()
    
    df = db.query(f"""
        SELECT 
            market,
            state,
            COUNT(*) as records,
            ROUND((MAX(arrival_date) - MIN(arrival_date))::float / 365.25, 1) as years
        FROM prices
        WHERE regexp_replace(commodity, '\\s+', ' ') = '{commodity.title()}'
        GROUP BY market, state
        HAVING COUNT(*) > 500
        ORDER BY records DESC
        LIMIT {limit}
    """)
    
    print(f"\n{'='*80}")
    print(f"Top {limit} states for {commodity} by historical coverage:")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")
