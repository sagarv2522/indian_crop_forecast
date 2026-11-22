import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("AZURE_BLOB_BASE_URL", "https://cropspricesstorage.blob.core.windows.net/crops-data").rstrip("/")
SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "").lstrip("?")
YEAR_START = int(os.getenv("AZURE_YEAR_START", "2016"))
YEAR_END = int(os.getenv("AZURE_YEAR_END", "2026"))

def parquet_url(year: int) -> str:
    token_str = f"?{SAS_TOKEN}" if SAS_TOKEN else ""
    return f"{BASE_URL}/data_{year}.parquet{token_str}"

def all_parquet_urls():
    return [
        parquet_url(year)
        for year in range(YEAR_START, YEAR_END + 1)
    ]
