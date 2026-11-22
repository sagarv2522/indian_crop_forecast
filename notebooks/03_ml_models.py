"""
notebooks/03_ml_models.py
──────────────────────────
Phase 3: XGBoost feature-based forecasting on a single crop-state series.

Run:
    python notebooks/03_ml_models.py --crop "Tomato" --state "Tamil Nadu"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.tabular_forecaster import XGBoostPriceForecaster
from src.utils.duckdb_client import DuckDBClient
from src.utils.market_selector import find_best_state, suggest_state


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("notebooks/ml_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_series(crop: str, state: str) -> pd.Series:
    db = DuckDBClient()
    return db.crop_series(crop, state, freq="D")


def main(crop: str, state: str, steps: int, test_size: int, auto_select_best: bool = True) -> None:
    # Auto-select state if the specified one is too sparse
    try:
        series = load_series(crop, state)
        if len(series) < 200 and auto_select_best:
            logger.warning(
                f"State '{state}' has only {len(series)} observations. "
                f"Auto-selecting state with better historical coverage..."
            )
            suggest_state(crop, limit=5)
            state = find_best_state(crop, min_records=1000, top_n=1)
            series = load_series(crop, state)
    except ValueError as e:
        logger.warning(f"Failed to load '{state}': {e}")
        logger.info(f"Auto-selecting best state for {crop}...")
        suggest_state(crop, limit=5)
        state = find_best_state(crop, min_records=1000, top_n=1)
        series = load_series(crop, state)

    # Print phase header AFTER auto-selection so it shows correct state
    print("\n" + "=" * 60)
    print("PHASE 3 — XGBOOST FEATURE MODEL")
    print(f"Crop: {crop} | State: {state}")
    print("=" * 60)
    
    model = XGBoostPriceForecaster(crop=crop, state=state)

    metrics = model.evaluate(series, test_size=test_size)
    print(f"\nHoldout metrics: {metrics}")

    model.fit(series)
    forecast = model.forecast(steps=steps)

    model_path = model.save()
    forecast_path = OUTPUT_DIR / f"xgb_forecast_{crop.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}.csv"
    forecast.to_csv(forecast_path, index=False)

    print(f"\nSaved model    : {model_path}")
    print(f"Saved forecast : {forecast_path}")

    fig, ax = plt.subplots(figsize=(13, 5))
    history = series.iloc[-180:]
    ax.plot(history.index, history.values, label="Actual", linewidth=1.2)
    ax.plot(forecast["date"], forecast["forecast"], label="Forecast", linestyle="--", linewidth=1.5)
    ax.axvline(history.index[-1], color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(f"XGBoost forecast: {crop} @ {state}")
    ax.set_ylabel("₹/quintal")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"xgb_forecast_{crop.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved chart   : {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: XGBoost model with auto state selection"
    )
    parser.add_argument("--crop", default="Tomato", help="Commodity name")
    parser.add_argument("--state", default="Tamil Nadu", help="State name (auto-selects best if too sparse)")
    parser.add_argument("--steps", type=int, default=30, help="Forecast steps")
    parser.add_argument("--test-size", type=int, default=90, help="Test set size")
    parser.add_argument("--no-auto-select", action="store_true", help="Disable auto-state selection")
    args = parser.parse_args()
    main(args.crop, args.state, args.steps, args.test_size, auto_select_best=not args.no_auto_select)
