"""Upload local partitioned Parquet files to Azure Blob Storage container 'crops-data'.

Maps local:  data/parquet/year=YYYY/data.parquet
To blob:    crops-data/data_YYYY.parquet
"""

import os
import re
from pathlib import Path
import duckdb
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "data" / "parquet"
CONTAINER_NAME = "crops-data"

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not CONN_STR:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set in .env")

# 1. Check for any root CSV files (e.g. crop_prices_2021.csv ... 2025.csv) that can be converted
for year in range(2021, 2026):
    csv_file = ROOT / f"crop_prices_{year}.csv"
    target_parquet = PARQUET_DIR / f"year={year}" / "data.parquet"
    if csv_file.exists() and not target_parquet.exists():
        print(f"Converting {csv_file.name} to {target_parquet}...")
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT 
                    CAST(arrival_date AS DATE) AS arrival_date,
                    state, district, market, commodity,
                    CAST(commodity_code AS VARCHAR) AS commodity_code,
                    variety, grade,
                    CAST(min_price AS FLOAT) AS min_price,
                    CAST(max_price AS FLOAT) AS max_price,
                    CAST(modal_price AS FLOAT) AS modal_price,
                    CAST(year AS SMALLINT) AS year,
                    CAST(month AS UTINYINT) AS month,
                    CAST(week AS UTINYINT) AS week,
                    CAST(quarter AS UTINYINT) AS quarter,
                    season
                FROM read_csv_auto('{csv_file.as_posix()}')
            ) TO '{target_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION 'snappy')
        """)
        print(f"  -> Generated {target_parquet.name} ({target_parquet.stat().st_size / 1024 / 1024:.1f} MB)")

# 2. Find all local Parquet partitions
parquet_files = sorted(PARQUET_DIR.glob("year=*/data.parquet"))
if not parquet_files:
    print(f"No parquet files found in {PARQUET_DIR}")
    exit(1)

print(f"\nConnecting to Azure Blob Storage ({CONTAINER_NAME})...")
blob_service = BlobServiceClient.from_connection_string(CONN_STR)
container_client = blob_service.get_container_client(CONTAINER_NAME)

if not container_client.exists():
    print(f"Container '{CONTAINER_NAME}' does not exist. Creating...")
    container_client.create_container()

print(f"\nUploading {len(parquet_files)} parquet file(s) to '{CONTAINER_NAME}':")
uploaded = []

for p in parquet_files:
    # Extract year from year=YYYY
    match = re.search(r"year=(\d{4})", str(p))
    if not match:
        continue
    year = match.group(1)
    blob_name = f"data_{year}.parquet"
    size_mb = p.stat().st_size / 1024 / 1024

    print(f"  Uploading {p.parent.name}/{p.name} ({size_mb:.2f} MB) -> {blob_name} ...", end="", flush=True)
    with open(p, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    print(" Done!")
    uploaded.append((blob_name, size_mb))

print("\n" + "=" * 60)
print(f"Successfully uploaded {len(uploaded)} file(s) to container '{CONTAINER_NAME}':")
for name, mb in uploaded:
    print(f"  - {name} ({mb:.2f} MB)")
print("=" * 60)
