"""
notebooks/01_eda.py
────────────────────
Phase 1 EDA — all analysis runs via DuckDB SQL on Parquet.
Pandas only receives small aggregated DataFrames for plotting.
80M rows, laptop RAM, no problem.

Run:
    python notebooks/01_eda.py
    python notebooks/01_eda.py --crop "Tomato"
    python notebooks/01_eda.py --crop "Tomato" --state "Tamil Nadu"
    python notebooks/01_eda.py --state "Tamil Nadu"

All charts  → notebooks/eda_outputs/
Summary CSV → notebooks/eda_outputs/summary_stats.csv
Series CSV  → notebooks/eda_outputs/series_<crop>_<market>.csv  (for Phase 2)
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.duckdb_client import DuckDBClient

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("notebooks/eda_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})
PRICE_FMT = mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}")


def save(name):
    plt.savefig(OUTPUT_DIR / name)
    plt.close()
    logger.info(f"Saved → {OUTPUT_DIR / name}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1  Dataset overview
# ═══════════════════════════════════════════════════════════════════
def section1_overview(db):
    print("\n" + "="*60)
    print("SECTION 1 — DATASET OVERVIEW")
    print("="*60)

    ov = db.overview()
    for k, v in ov.items():
        print(f"  {k:<22}: {v}")

    yearly = db.query("""
        SELECT year, COUNT(*) AS records
        FROM prices
        GROUP BY year ORDER BY year
    """)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(yearly["year"], yearly["records"] / 1_000,
           color="#378ADD", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Records (thousands)")
    ax.set_title("Records per year — 2016 to 2026")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    save("01_records_per_year.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 2  Data quality audit
# ═══════════════════════════════════════════════════════════════════
def section2_quality(db):
    print("\n" + "="*60)
    print("SECTION 2 — DATA QUALITY AUDIT")
    print("="*60)

    q = db.query("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN modal_price IS NULL   THEN 1 ELSE 0 END) AS null_modal,
            SUM(CASE WHEN min_price   IS NULL   THEN 1 ELSE 0 END) AS null_min,
            SUM(CASE WHEN max_price   IS NULL   THEN 1 ELSE 0 END) AS null_max,
            SUM(CASE WHEN state       IS NULL   THEN 1 ELSE 0 END) AS null_state,
            SUM(CASE WHEN min_price > modal_price
                     THEN 1 ELSE 0 END) AS min_gt_modal,
            SUM(CASE WHEN modal_price > max_price
                     THEN 1 ELSE 0 END) AS modal_gt_max
        FROM prices
    """)
    total = int(q["total"].iloc[0])
    checks = {
        "Null modal_price"  : int(q["null_modal"].iloc[0]),
        "Null min_price"    : int(q["null_min"].iloc[0]),
        "Null max_price"    : int(q["null_max"].iloc[0]),
        "Null state"        : int(q["null_state"].iloc[0]),
        "Min > Modal (bad)" : int(q["min_gt_modal"].iloc[0]),
        "Modal > Max (bad)" : int(q["modal_gt_max"].iloc[0]),
    }
    print(f"\n  {'Check':<25} {'Count':>10}  {'%':>8}")
    print("  " + "─"*47)
    for k, v in checks.items():
        flag = " ✓" if v == 0 else " ⚠"
        print(f"  {k:<25} {v:>10,}  {v/total*100:>7.3f}%{flag}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 3  Price distributions
# ═══════════════════════════════════════════════════════════════════
def section3_distributions(db):
    print("\n" + "="*60)
    print("SECTION 3 — PRICE DISTRIBUTIONS")
    print("="*60)

    stats = db.query("""
        SELECT
            ROUND(MIN(modal_price),   2) AS min,
            ROUND(MEDIAN(modal_price),2) AS median,
            ROUND(AVG(modal_price),   2) AS mean,
            ROUND(MAX(modal_price),   2) AS max,
            ROUND(STDDEV(modal_price),2) AS std,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP
                (ORDER BY modal_price), 2) AS p5,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                (ORDER BY modal_price), 2) AS p95
        FROM prices
    """)
    print(f"\n  modal_price stats (₹/quintal):")
    for col in stats.columns:
        print(f"    {col:>8} : ₹{stats[col].iloc[0]:>12,.2f}")

    hist = db.query("""
        SELECT
            FLOOR(modal_price / 1000) * 1000 AS bucket,
            COUNT(*) AS cnt
        FROM prices
        WHERE modal_price < 30000
        GROUP BY bucket ORDER BY bucket
    """)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(hist["bucket"], hist["cnt"] / 1_000,
           width=900, color="#5DCAA5", alpha=0.85,
           edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Modal price (₹/quintal)")
    ax.set_ylabel("Records (thousands)")
    ax.set_title("Modal price distribution — all crops, all states (2016–2026)")
    ax.xaxis.set_major_formatter(PRICE_FMT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    save("03_price_distribution.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 4  Seasonal patterns
# ═══════════════════════════════════════════════════════════════════
def section4_seasonal(db, commodity=None):
    print("\n" + "="*60)
    print("SECTION 4 — SEASONAL PATTERNS")
    print("="*60)

    monthly = db.query("""
        SELECT month,
               ROUND(MEDIAN(modal_price), 2) AS median_price,
               ROUND(STDDEV(modal_price),  2) AS std_price
        FROM prices
        GROUP BY month ORDER BY month
    """)

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly["month"], monthly["median_price"],
           color="#534AB7", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.errorbar(monthly["month"], monthly["median_price"],
                yerr=monthly["std_price"] * 0.3,
                fmt="none", color="#0C447C", capsize=3, linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_ylabel("Median modal price (₹/quintal)")
    ax.set_title("Monthly price seasonality — all crops combined")
    ax.yaxis.set_major_formatter(PRICE_FMT)
    save("04a_monthly_seasonality.png")

    if commodity:
        crop_s = db.seasonal_avg(commodity)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(crop_s["month"], crop_s["median_price"],
                color="#D85A30", marker="o", linewidth=2, markersize=6)
        ax.fill_between(
            crop_s["month"],
            crop_s["median_price"] - crop_s["price_stddev"] * 0.3,
            crop_s["median_price"] + crop_s["price_stddev"] * 0.3,
            alpha=0.15, color="#D85A30"
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.set_ylabel("Median modal price (₹/quintal)")
        ax.set_title(f"Monthly seasonality — {commodity}")
        ax.yaxis.set_major_formatter(PRICE_FMT)
        save(f"04c_seasonality_{commodity.lower().replace(' ', '_')}.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5  State-wise analysis
# ═══════════════════════════════════════════════════════════════════
def section5_states(db):
    print("\n" + "="*60)
    print("SECTION 5 — STATE-WISE ANALYSIS")
    print("="*60)

    state_df = db.state_summary()
    print(f"\n  Top 10 states by record volume:\n")
    print(state_df.head(10).to_string(index=False))

    top15 = state_df.head(15).sort_values("median_price")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top15["state"], top15["median_price"],
            color="#185FA5", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Median modal price (₹/quintal)")
    ax.set_title("State median crop price — top 15 states by data volume")
    ax.xaxis.set_major_formatter(PRICE_FMT)
    save("05_state_prices.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 6  Top crops
# ═══════════════════════════════════════════════════════════════════
def section6_crops(db):
    print("\n" + "="*60)
    print("SECTION 6 — TOP CROPS BY DATA VOLUME")
    print("="*60)

    crops = db.list_crops(min_records=1000)
    print(f"\n  Total commodities (1000+ records): {len(crops)}")
    print(crops.head(15).to_string(index=False))

    top20 = crops.head(20).sort_values("records")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top20["commodity"], top20["records"] / 1000,
            color="#1D9E75", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Records (thousands)")
    ax.set_title("Top 20 commodities by data volume (2016–2026)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    save("06_top_crops.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 7  Year-on-year trend
# ═══════════════════════════════════════════════════════════════════
def section7_trend(db, commodity=None):
    print("\n" + "="*60)
    print("SECTION 7 — YEAR-ON-YEAR PRICE TREND")
    print("="*60)

    if not commodity:
        commodity = db.scalar("""
            SELECT commodity FROM prices
            GROUP BY commodity ORDER BY COUNT(*) DESC LIMIT 1
        """)
    print(f"\n  Crop: {commodity}")

    yearly = db.yearly_trend(commodity)
    print(yearly.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yearly["year"], yearly["median_price"],
            color="#378ADD", marker="o", linewidth=2.2,
            markersize=7, zorder=3)
    ax.fill_between(yearly["year"], yearly["median_price"],
                    alpha=0.12, color="#378ADD")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median modal price (₹/quintal)")
    ax.set_title(f"Year-on-year price trend — {commodity}")
    ax.yaxis.set_major_formatter(PRICE_FMT)
    ax.set_xticks(yearly["year"])
    save(f"07_yearly_trend_{commodity.lower().replace(' ', '_')}.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 8  Time series (weekly) — feeds directly into Phase 2
# ═══════════════════════════════════════════════════════════════════
def section8_timeseries(db, commodity,state):
    print("\n" + "="*60)
    print("SECTION 8 — TIME SERIES (feeds Phase 2)")
    print("="*60)

    try:
        series = db.crop_series(commodity, state, freq="W")
    except ValueError as e:
        logger.warning(str(e))
        fallback = db.list_state(commodity).iloc[0]["state"]
        logger.info(f"Fallback state: {fallback}")
        series = db.crop_series(commodity, fallback, freq="W")
        state = fallback

    rolling = series.rolling(12).mean()

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(series.index, series.values,
            color="#B5D4F4", linewidth=0.8, alpha=0.8, label="Weekly median")
    ax.plot(rolling.index, rolling.values,
            color="#185FA5", linewidth=1.8, label="12-week rolling avg")
    ax.set_ylabel("₹/quintal")
    ax.set_title(f"Price time series — {commodity} @ {state}")
    ax.yaxis.set_major_formatter(PRICE_FMT)
    ax.legend()
    save(f"08_timeseries_{commodity.lower().replace(' ','_')}.png")

    print(f"\n  Series: {len(series)} weekly points")
    print(f"  Range : {series.index[0].date()} → {series.index[-1].date()}")
    print(f"  Mean  : ₹{series.mean():,.2f}/quintal")
    print(f"\n  ✓ Series ready for Phase 2")

    csv_name = f"series_{commodity.lower().replace(' ','_')}_{state.lower().replace(' ','_')[:20]}.csv"
    series.to_csv(OUTPUT_DIR / csv_name, header=True)
    logger.info(f"Series saved → {OUTPUT_DIR / csv_name}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 9  Market volatility
# ═══════════════════════════════════════════════════════════════════
def section9_volatility(db, commodity):
    print("\n" + "="*60)
    print("SECTION 9 — MARKET PRICE VOLATILITY")
    print("="*60)

    vol = db.top_volatile_markets(commodity, top_n=15)
    print(f"\n  Most volatile markets for {commodity}:\n")
    print(vol.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r['market']} ({r['state']})" for _, r in vol.iterrows()]
    ax.barh(labels, vol["volatility"],
            color="#EF9F27", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Price std dev (₹/quintal)")
    ax.set_title(f"Market price volatility — {commodity}")
    ax.xaxis.set_major_formatter(PRICE_FMT)
    save(f"09_volatility_{commodity.lower().replace(' ','_')}.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 10  Export summary
# ═══════════════════════════════════════════════════════════════════
def section10_export(db):
    print("\n" + "="*60)
    print("SECTION 10 — EXPORT SUMMARY STATS")
    print("="*60)

    summary = db.query("""
        SELECT
            commodity, state,
            COUNT(*) AS records,
            ROUND(MIN(modal_price),    2) AS min_price,
            ROUND(MEDIAN(modal_price), 2) AS median_price,
            ROUND(MAX(modal_price),    2) AS max_price,
            ROUND(AVG(modal_price),    2) AS mean_price,
            ROUND(STDDEV(modal_price), 2) AS std_price,
            ROUND(STDDEV(modal_price) / AVG(modal_price), 3) AS cv
        FROM prices
        GROUP BY commodity, state
        HAVING COUNT(*) >= 365
        ORDER BY records DESC
    """)

    out = OUTPUT_DIR / "summary_stats.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Exported {len(summary):,} commodity-state pairs → {out}")
    print(summary.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main(commodity, state):
    print("\n" + "="*60)
    print("PHASE 1 — EDA  (DuckDB + Parquet)")
    print(f"Crop: {commodity or 'ALL'}  |  State: {state or 'AUTO'}")
    print("="*60)

    with DuckDBClient() as db:
        section1_overview(db)
        section2_quality(db)
        section3_distributions(db)
        section4_seasonal(db, commodity)
        section5_states(db)
        section6_crops(db)
        section7_trend(db, commodity)
        section8_timeseries(db, commodity or "Tomato", state or "Tamil Nadu")
        section9_volatility(db, commodity or "Tomato")
        section10_export(db)

    print(f"\n{'='*60}")
    print(f"EDA COMPLETE — charts + CSV → {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop",   default="Tomato",    help="Primary crop to analyse")
    parser.add_argument("--state",  default="Tamil Nadu",       help="Market for time-series")
    args = parser.parse_args()
    main(args.crop, args.state)
