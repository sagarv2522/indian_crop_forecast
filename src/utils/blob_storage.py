import os

try:
    import streamlit as st

    BASE_URL = st.secrets["AZURE_BLOB_BASE_URL"]
    SAS_TOKEN = st.secrets["AZURE_SAS_TOKEN"]

except Exception:
    BASE_URL = os.getenv("AZURE_BLOB_BASE_URL")
    SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")


def parquet_url(year: int) -> str:
    return f"{BASE_URL}/data_{year}.parquet?{SAS_TOKEN}"


def all_parquet_urls():
    return [
        parquet_url(year)
        for year in range(2016, 2027)
    ]