import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("AZURE_BLOB_BASE_URL", "").rstrip("/")
SAS = os.getenv("AZURE_SAS_TOKEN", "").lstrip("?")

if not (BASE_URL and SAS):
    raise ValueError("AZURE_BLOB_BASE_URL and AZURE_SAS_TOKEN must be set in .env")

url = f"{BASE_URL}/data_2026.parquet?{SAS}"

con = duckdb.connect()

df = con.execute(f"""
SELECT *
FROM read_parquet('{url}')
LIMIT 5
""").fetchdf()

print(df)
