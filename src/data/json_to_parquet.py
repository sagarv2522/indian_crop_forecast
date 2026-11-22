"""
src/data/json_to_parquet.py
────────────────────────────
Converts all raw JSON files (data/raw/YYYY_MM/page_*.json)
into partitioned Parquet files (data/parquet/year=YYYY/).

Why Parquet over CSV:
  - 80M rows as CSV  ≈ 8–10 GB  (pandas would die loading it)
  - 80M rows Parquet ≈ 1.5–2 GB (columnar — reads only queried columns)
  - DuckDB queries Parquet in seconds without ever loading full dataset

Raw JSON schema (confirmed from live API):
  Arrival_Date   str  DD/MM/YYYY
  Commodity      str
  Commodity_Code str
  District       str
  Grade          str
  Market         str
  Max_Price      str  (numeric stored as string)
  Min_Price      str
  Modal_Price    str
  State          str
  Variety        str

Run:
    python -m src.data.json_to_parquet
    python -m src.data.json_to_parquet --resume   (skip already done years)
    python -m src.data.json_to_parquet --year 2022 (single year only)
    python -m src.data.json_to_parquet --csv     (write crop_price_2016.csv ... crop_price_2020.csv)
"""

import re
import json
import logging
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
PARQUET_DIR = Path("data/parquet")
CSV_DIR     = Path("data/csv")

# ── Date range of our download ─────────────────────────────────────────────
START_YEAR = 2016
END_YEAR   = 2026
CSV_END_YEAR = 2020

# ── Price sanity bounds (₹ / quintal) ─────────────────────────────────────
PRICE_MIN =     1.0
PRICE_MAX = 200_000.0

# ── Season map ────────────────────────────────────────────────────────────
SEASON_MAP = {
    1: "Rabi", 2: "Rabi",  3: "Rabi",  4: "Rabi",
    5: "Zaid", 6: "Zaid",
    7: "Kharif", 8: "Kharif", 9: "Kharif",
    10: "Kharif", 11: "Kharif", 12: "Rabi",
}

# ── PyArrow output schema ─────────────────────────────────────────────────
PARQUET_SCHEMA = pa.schema([
    ("arrival_date",   pa.date32()),
    ("state",          pa.string()),
    ("district",       pa.string()),
    ("market",         pa.string()),
    ("commodity",      pa.string()),
    ("commodity_code", pa.string()),
    ("variety",        pa.string()),
    ("grade",          pa.string()),
    ("min_price",      pa.float32()),
    ("max_price",      pa.float32()),
    ("modal_price",    pa.float32()),
    ("year",           pa.int16()),
    ("month",          pa.int8()),
    ("week",           pa.int8()),
    ("quarter",        pa.int8()),
    ("season",         pa.string()),
])


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Discover all raw JSON files grouped by year
# ─────────────────────────────────────────────────────────────────────────────

def iter_year_groups(
    target_year: int = None,
) -> Iterator[tuple[int, list[Path]]]:
    """
    Yields (year, [list of page_*.json paths]) for every YYYY_MM folder.
    Sorted chronologically. Optionally filtered to a single year.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    # Collect all YYYY_MM folders
    folder_pattern = re.compile(r"^(\d{4})_(\d{2})$")
    year_files: dict[int, list[Path]] = {}

    for folder in sorted(RAW_DIR.iterdir()):
        m = folder_pattern.match(folder.name)
        if not m or not folder.is_dir():
            continue
        year = int(m.group(1))
        if target_year and year != target_year:
            continue
        pages = sorted(folder.glob("page_*.json"))
        if pages:
            year_files.setdefault(year, []).extend(pages)

    for year in sorted(year_files):
        yield year, year_files[year]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load records from one JSON page file
# ─────────────────────────────────────────────────────────────────────────────

def load_json_page(path: Path) -> list[dict]:
    """Read one page_*.json and return the records list."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("records", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"  Skipping {path.name}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Clean & type-cast a batch of raw records
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    """Strip + title-case + collapse all internal whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().title())


def clean_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert raw API record dicts into a clean typed DataFrame.

    Transformations:
      - Rename API field names → snake_case
      - Arrival_Date (DD/MM/YYYY str) → datetime.date
      - Price strings → float32  (commas removed, outliers dropped)
      - Text fields  → normalized (strip, title-case, collapse whitespace)
      - Add: year, month, week, quarter, season
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Rename columns ────────────────────────────────────────────────────
    df = df.rename(columns={
        "Arrival_Date":   "arrival_date",
        "Commodity":      "commodity",
        "Commodity_Code": "commodity_code",
        "District":       "district",
        "Grade":          "grade",
        "Market":         "market",
        "Max_Price":      "max_price",
        "Min_Price":      "min_price",
        "Modal_Price":    "modal_price",
        "State":          "state",
        "Variety":        "variety",
    })

    # ── Parse dates (DD/MM/YYYY) ──────────────────────────────────────────
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"], dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["arrival_date"])

    # ── Prices: remove commas, coerce to float, drop invalid ─────────────
    for col in ("min_price", "max_price", "modal_price"):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
            .astype("float32")
        )

    df = df.dropna(subset=["modal_price"])
    df = df[df["modal_price"].between(PRICE_MIN, PRICE_MAX)]

    # min ≤ modal ≤ max  — drop physically impossible rows
    if "min_price" in df.columns and "max_price" in df.columns:
        df = df[
            (df["min_price"] <= df["modal_price"]) &
            (df["modal_price"] <= df["max_price"])
        ]

    # ── Force commodity_code to string (API returns int in some months) ──
    if "commodity_code" in df.columns:
        df["commodity_code"] = df["commodity_code"].astype(str).str.strip()

    # ── Normalize text ────────────────────────────────────────────────────
    for col in ("state", "district", "market", "commodity", "variety", "grade"):
        if col in df.columns:
            df[col] = df[col].apply(_normalize_text)
    df["variety"] = df["variety"].replace("Nan", "Other")

    # ── Time features ─────────────────────────────────────────────────────
    df["year"]    = df["arrival_date"].dt.year.astype("int16")
    df["month"]   = df["arrival_date"].dt.month.astype("int8")
    df["week"]    = df["arrival_date"].dt.isocalendar().week.astype("int8")
    df["quarter"] = df["arrival_date"].dt.quarter.astype("int8")
    df["season"]  = df["month"].map(SEASON_MAP)

    # ── Convert arrival_date to date (drop time component) ───────────────
    df["arrival_date"] = df["arrival_date"].dt.date

    return df[[
        "arrival_date", "state", "district", "market",
        "commodity", "commodity_code", "variety", "grade",
        "min_price", "max_price", "modal_price",
        "year", "month", "week", "quarter", "season",
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Write one year's data to Parquet
# ─────────────────────────────────────────────────────────────────────────────

def write_year_parquet(year: int, df: pd.DataFrame) -> Path:
    """
    Write a year's DataFrame to data/parquet/year=YYYY/data.parquet.
    Uses hive-style partitioning so DuckDB can filter by year without
    reading other partition files.
    """
    out_dir  = PARQUET_DIR / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"

    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(
        table,
        out_path,
        compression="snappy",       # fast read/write, good compression
        use_dictionary=True,        # great for low-cardinality string cols
        write_statistics=True,      # enables predicate pushdown in DuckDB
    )
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info(f"  → {out_path}  ({len(df):,} rows, {size_mb:.1f} MB)")
    return out_path


def write_year_csv(year: int, df: pd.DataFrame) -> Path:
    """Write one year's raw combined data to data/csv/crop_price_YYYY.csv."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CSV_DIR / f"crop_price_{year}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info(f"  → {out_path}  ({len(df):,} rows, {size_mb:.1f} MB)")
    return out_path


def convert_all_csv(
    start_year: int = START_YEAR,
    end_year: int = CSV_END_YEAR,
    resume: bool = False,
) -> None:
    """Combine monthly raw JSON files into one CSV file per year."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_files = 0
    skipped_years = 0

    for year, pages in iter_year_groups():
        if year < start_year or year > end_year:
            continue

        out_path = CSV_DIR / f"crop_price_{year}.csv"
        if resume and out_path.exists():
            logger.info(f"[{year}] Skipping — already converted ({out_path})")
            skipped_years += 1
            continue

        logger.info(f"[{year}] Combining {len(pages)} page files...")
        year_frames = []
        for page_path in pages:
            records = load_json_page(page_path)
            if records:
                year_frames.append(pd.DataFrame(records))

        if not year_frames:
            logger.warning(f"[{year}] No records found — skipping")
            continue

        df_year = (
            pd.concat(year_frames, ignore_index=True)
            .reset_index(drop=True)
        )
        write_year_csv(year, df_year)
        total_rows += len(df_year)
        total_files += len(pages)
        logger.info(f"[{year}] Done — {len(df_year):,} raw rows from {len(pages)} pages")

    logger.info(
        f"\n{'='*55}\n"
        f"CSV conversion complete\n"
        f"  Total rows written : {total_rows:,}\n"
        f"  JSON files read    : {total_files:,}\n"
        f"  Years skipped      : {skipped_years}\n"
        f"  Output             : {CSV_DIR.resolve()}\n"
        f"{'='*55}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def convert_all(resume: bool = False, target_year: int = None) -> None:
    """
    Full pipeline: walk all raw JSON folders → clean → write Parquet.

    Args:
        resume      : skip years that already have a Parquet file
        target_year : process only this year (None = all years)
    """
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    total_rows   = 0
    total_files  = 0
    skipped_years = 0

    for year, pages in iter_year_groups(target_year):
        out_path = PARQUET_DIR / f"year={year}" / "data.parquet"

        if resume and out_path.exists():
            logger.info(f"[{year}] Skipping — already converted ({out_path})")
            skipped_years += 1
            continue

        logger.info(f"[{year}] Processing {len(pages)} page files...")

        year_frames = []
        for page_path in pages:
            records = load_json_page(page_path)
            if records:
                df_page = clean_records(records)
                if not df_page.empty:
                    year_frames.append(df_page)

        if not year_frames:
            logger.warning(f"[{year}] No valid records found — skipping")
            continue

        df_year = (
            pd.concat(year_frames, ignore_index=True)
            .drop_duplicates(subset=["commodity", "market", "state", "arrival_date"])
            .sort_values("arrival_date")
            .reset_index(drop=True)
        )

        write_year_parquet(year, df_year)
        total_rows  += len(df_year)
        total_files += len(pages)

        logger.info(
            f"[{year}] Done — {len(df_year):,} clean rows "
            f"from {len(pages)} pages"
        )

    logger.info(
        f"\n{'='*55}\n"
        f"Conversion complete\n"
        f"  Total rows written : {total_rows:,}\n"
        f"  JSON files read    : {total_files:,}\n"
        f"  Years skipped      : {skipped_years}\n"
        f"  Output             : {PARQUET_DIR.resolve()}\n"
        f"{'='*55}"
    )


def print_summary() -> None:
    """Print a quick summary of what's already been converted."""
    parts = sorted(PARQUET_DIR.glob("year=*/data.parquet"))
    if not parts:
        print("No Parquet files found. Run without --summary first.")
        return
    total_size = 0
    total_rows = 0
    print(f"\n{'Year':<8} {'Rows':>12} {'Size MB':>10}")
    print("─" * 32)
    for p in parts:
        year = p.parent.name.split("=")[1]
        size = p.stat().st_size / 1_048_576
        rows = pq.read_metadata(p).num_rows
        total_size += size
        total_rows += rows
        print(f"{year:<8} {rows:>12,} {size:>10.1f}")
    print("─" * 32)
    print(f"{'TOTAL':<8} {total_rows:>12,} {total_size:>10.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw JSON downloads to partitioned Parquet"
    )
    parser.add_argument(
        "--resume",  action="store_true",
        help="Skip years that already have a Parquet file"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Process a single year only (e.g. --year 2022)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary of already-converted files and exit"
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Write one CSV per year for 2016 through 2020"
    )
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.csv:
        if args.year is not None:
            convert_all_csv(start_year=args.year, end_year=args.year, resume=args.resume)
        else:
            convert_all_csv(resume=args.resume)
    else:
        convert_all(resume=args.resume, target_year=args.year)
