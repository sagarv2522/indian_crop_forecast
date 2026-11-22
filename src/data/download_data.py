from pathlib import Path
from datetime import datetime
import calendar
import subprocess
import json
import math
import argparse
import os

# ==========================================================
# CONFIG
# ==========================================================

API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "").strip()

RESOURCE_ID = os.getenv(
    "DATA_GOV_IN_RESOURCE_ID",
    "35985678-0d79-46b4-9ed6-6f13308a1d24"
).strip()

BASE_URL = (
    f"https://api.data.gov.in/resource/{RESOURCE_ID}"
)

LIMIT = 10000

START_YEAR = 2016
START_MONTH = 1

TODAY = datetime.now()
END_YEAR = TODAY.year
END_MONTH = TODAY.month

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# MONTH GENERATOR
# ==========================================================

def generate_months(
    start_year,
    start_month,
    end_year,
    end_month
):
    year = start_year
    month = start_month

    while (
        year < end_year
        or
        (
            year == end_year
            and month <= end_month
        )
    ):
        yield year, month

        month += 1

        if month > 12:
            month = 1
            year += 1

# ==========================================================
# MONTH DATE RANGE
# ==========================================================

def get_month_range(year, month):

    first_day = (
        f"{year}-{month:02d}-01"
    )

    last_day_num = calendar.monthrange(
        year,
        month
    )[1]

    last_day = (
        f"{year}-{month:02d}-{last_day_num:02d}"
    )

    return first_day, last_day

# ==========================================================
# CURL REQUEST
# ==========================================================

def fetch_page(
    start_date,
    end_date,
    offset
):
    if not API_KEY:
        raise RuntimeError(
            "DATA_GOV_IN_API_KEY is not set. Export your data.gov.in API key "
            "before running the downloader."
        )

    cmd = [
        "curl",
        "-s",
        "-G",
        BASE_URL,

        "--data-urlencode",
        f"api-key={API_KEY}",

        "--data-urlencode",
        "format=json",

        "--data-urlencode",
        f"limit={LIMIT}",

        "--data-urlencode",
        f"offset={offset}",

        "--data-urlencode",
        f"range[Arrival_Date][gte]={start_date}",

        "--data-urlencode",
        f"range[Arrival_Date][lte]={end_date}",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Curl failed:\n{result.stderr}"
        )

    return json.loads(result.stdout)

# ==========================================================
# SAVE PAGE
# ==========================================================

def save_page(
    data,
    month_folder,
    offset
):

    output_file = (
        month_folder /
        f"page_{offset:07d}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

# ==========================================================
# DOWNLOAD MONTH
# ==========================================================

def download_month(year, month):

    print(
        f"\nDownloading "
        f"{year}-{month:02d}"
    )

    start_date, end_date = (
        get_month_range(year, month)
    )

    month_folder = (
        RAW_DIR /
        f"{year}_{month:02d}"
    )

    month_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------
    # First Page
    # -----------------------

    first_page = fetch_page(
        start_date,
        end_date,
        offset=0
    )

    total = int(first_page["total"])
    count = int(first_page["count"])

    print(
        f"Total Records: {total:,}"
    )

    if total == 0:
        print("No data")
        return

    save_page(
        first_page,
        month_folder,
        offset=0
    )

    pages = math.ceil(
        total / LIMIT
    )

    print(
        f"Pages Required: {pages}"
    )

    # -----------------------
    # Remaining Pages
    # -----------------------

    for page in range(1, pages):

        offset = page * LIMIT

        output_file = (
            month_folder /
            f"page_{offset:07d}.json"
        )

        # Resume support
        if output_file.exists():

            print(
                f"SKIP "
                f"offset={offset}"
            )

            continue

        data = fetch_page(
            start_date,
            end_date,
            offset
        )

        save_page(
            data,
            month_folder,
            offset
        )

        print(
            f"Saved "
            f"offset={offset:,}"
        )

# ==========================================================
# MAIN
# ==========================================================

def main(start_year, start_month, end_year, end_month):

    start = datetime.now()

    for year, month in generate_months(
        start_year,
        start_month,
        end_year,
        end_month
    ):
        download_month(
            year,
            month
        )

    end = datetime.now()

    print("\nCompleted")
    print(
        "Duration:",
        end - start
    )

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download monthly mandi price JSON dumps from data.gov.in"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"First year to download (default: {START_YEAR})",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=START_MONTH,
        help=f"First month to download, 1-12 (default: {START_MONTH})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"Last year to download (default: {END_YEAR})",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=END_MONTH,
        help=f"Last month to download, 1-12 (default: {END_MONTH})",
    )
    args = parser.parse_args()
    main(args.start_year, args.start_month, args.end_year, args.end_month)
