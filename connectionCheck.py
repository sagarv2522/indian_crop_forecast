import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Azure Blob Config (from .env)
# ============================================================
BASE_URL = os.getenv("AZURE_BLOB_BASE_URL", "").rstrip("/")
SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "").lstrip("?")
YEAR_START = int(os.getenv("AZURE_YEAR_START", "2021"))
YEAR_END = int(os.getenv("AZURE_YEAR_END", "2026"))

if not (BASE_URL and SAS_TOKEN):
    raise ValueError("AZURE_BLOB_BASE_URL and AZURE_SAS_TOKEN must be configured in .env")

print("=" * 60)
print("DuckDB Version:", duckdb.__version__)
print("Connecting to:", BASE_URL)
print(f"Year Range: {YEAR_START} - {YEAR_END}")
print("=" * 60)

con = duckdb.connect()

# ============================================================
# TEST 1 - Read sample rows from one file
# ============================================================
test_year = 2024
single_file = f"{BASE_URL}/data_{test_year}.parquet?{SAS_TOKEN}"
print(f"\nTEST 1: Reading sample rows from data_{test_year}.parquet")

try:
    sample_df = con.execute(f"""
        SELECT *
        FROM read_parquet('{single_file}')
        LIMIT 5
    """).fetchdf()

    print("\nSUCCESS - Sample Data:")
    print(sample_df)

except Exception as e:
    print(f"\nFAILED - Cannot read data_{test_year}.parquet")
    print(e)
    raise SystemExit

# ============================================================
# TEST 2 - Count rows in one file
# ============================================================
print("\n" + "=" * 60)
print(f"TEST 2: Counting rows in data_{test_year}.parquet")
print("=" * 60)

try:
    row_count = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{single_file}')
    """).fetchone()[0]

    print(f"Rows in data_{test_year}.parquet: {row_count:,}")

except Exception as e:
    print("\nFAILED - Cannot count rows")
    print(e)
    raise SystemExit

# ============================================================
# TEST 3 - Count rows across ALL yearly files
# ============================================================
print("\n" + "=" * 60)
print(f"TEST 3: Reading ALL yearly parquet files ({YEAR_START} to {YEAR_END})")
print("=" * 60)

urls = [
    f"{BASE_URL}/data_{year}.parquet?{SAS_TOKEN}"
    for year in range(YEAR_START, YEAR_END + 1)
]

url_list = ",".join([f"'{u}'" for u in urls])

query = f"""
SELECT COUNT(*)
FROM read_parquet([{url_list}])
"""

try:
    total_rows = con.execute(query).fetchone()[0]

    print(f"\nSUCCESS")
    print(f"Total Rows Across All Files ({YEAR_START}–{YEAR_END}): {total_rows:,}")

except Exception as e:
    print("\nFAILED - Cannot read all files")
    print(e)

# ============================================================
# TEST 4 - Query Example
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Example Query (Top Commodities)")
print("=" * 60)

query = f"""
SELECT
    commodity,
    COUNT(*) AS records
FROM read_parquet([{url_list}])
GROUP BY commodity
ORDER BY records DESC
LIMIT 10
"""

try:
    result = con.execute(query).fetchdf()

    print("\nTop 10 Commodities:")
    print(result)

except Exception as e:
    print("\nFAILED - Example query")
    print(e)

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
