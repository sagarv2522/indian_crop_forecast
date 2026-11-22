import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

print("DuckDB Version:", duckdb.__version__)

con = duckdb.connect()

con.execute("INSTALL azure")
con.execute("LOAD azure")

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT", "cropstorageiejray")
SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "").lstrip("?")

if CONN_STR:
    con.execute(f"""
    CREATE OR REPLACE SECRET azure_secret (
        TYPE AZURE,
        CONNECTION_STRING '{CONN_STR}'
    )
    """)
    print("Secret created from AZURE_STORAGE_CONNECTION_STRING")

url = f"https://{ACCOUNT_NAME}.blob.core.windows.net/crops-data/data_2024.parquet"
if SAS_TOKEN:
    url = f"{url}?{SAS_TOKEN}"

df = con.execute(f"""
SELECT *
FROM read_parquet('{url}')
LIMIT 5
""").fetchdf()

print(df.head())
