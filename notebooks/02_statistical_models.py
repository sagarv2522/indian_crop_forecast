"""
notebooks/02_statistical_models.py
────────────────────────────────────
Phase 2: Statistical Modelling — complete analysis notebook.
Runs all 8 steps end-to-end for one crop-state pair,
then saves the baseline model and CV scores.

Run:
    python notebooks/02_statistical_models.py
    python notebooks/02_statistical_models.py --crop "Tomato" --state "Tamil Nadu"
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

# local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.arima_model import ARIMAForecaster
from src.utils.duckdb_client import DuckDBClient
from src.utils.market_selector import find_best_state, suggest_state

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

OUTPUT_DIR = Path("notebooks/statistical_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DATA = "data/processed/clean_prices.csv"
PARQUET_DIR = Path("data/parquet")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def load_series_from_csv(df: pd.DataFrame, crop: str, state: str) -> pd.Series:
    """Extract daily modal_price series for one crop-state pair."""

    mask = (
        df["crop"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(crop.strip().lower())
    ) & (
        df["state"]
        .astype(str)
        .str.lower()
        .str.contains(
            state.strip().lower(),
            na=False,
            regex=False
        )
    )

    sub = df[mask].copy()

    print(f"\nMatched rows: {len(sub)}")

    if sub.empty:
        crop_matches = df[
            df["crop"].astype(str).str.contains(
                crop,
                case=False,
                na=False,
                regex=False
            )
        ]

        print(f"\nFound {len(crop_matches)} rows containing '{crop}'")

        if len(crop_matches):
            print("\nMatching crop names:")
            print(sorted(crop_matches["crop"].unique()))

            print("\nMarkets for those crops:")
            print(sorted(crop_matches["state"].unique())[:100])

        raise ValueError(
            f"No data found for crop='{crop}', state='{state}'"
        )

    series = (
        sub.groupby("arrival_date")["modal_price"]
        .median()
        .asfreq("D")
        .ffill()
    )

    logger.info(
        f"Loaded series: {crop} @ {state} — "
        f"{len(series)} days "
        f"({series.index[0].date()} → {series.index[-1].date()})"
    )

    return series


def load_series(crop: str, state: str) -> pd.Series:
    """
    Load crop-state series from Azure Blob via DuckDB.
    """

    logger.info("Loading series from DuckDB")

    db = DuckDBClient()

    return db.crop_series(
        crop,
        state,
        freq="D"
    )


def save_fig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Load & visualise raw series
# ═══════════════════════════════════════════════════════════════════
def step1_raw_series(series: pd.Series, crop: str, state: str) -> None:
    print("\n" + "="*60)
    print("STEP 1: RAW PRICE SERIES")
    print("="*60)
    print(f"\nSeries stats:\n{series.describe().round(2)}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Raw price
    axes[0].plot(series, color="#378ADD", linewidth=0.8, alpha=0.7, label="Daily price")
    axes[0].plot(series.rolling(30).mean(), color="#0C447C", linewidth=1.8, label="30-day MA")
    axes[0].set_title(f"{crop} @ {state} — Modal price (₹/quintal)")
    axes[0].set_ylabel("₹/quintal")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    axes[0].legend()

    # Log-transformed price
    log_series = np.log(series)
    axes[1].plot(log_series, color="#1D9E75", linewidth=0.8, alpha=0.7, label="Log price")
    axes[1].plot(log_series.rolling(30).mean(), color="#085041", linewidth=1.8, label="30-day MA")
    axes[1].set_title("Log-transformed price — variance stabilisation check")
    axes[1].set_ylabel("log(₹/quintal)")
    axes[1].legend()

    plt.suptitle("Step 1 — Raw series overview", fontsize=13)
    plt.tight_layout()
    save_fig("01_raw_series.png")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Stationarity tests
# ═══════════════════════════════════════════════════════════════════
def step2_stationarity(series: pd.Series) -> tuple:
    print("\n" + "="*60)
    print("STEP 2: STATIONARITY TESTS")
    print("="*60)

    forecaster = ARIMAForecaster()
    is_stationary, d = forecaster.check_stationarity(series)

    # Plot original vs differenced
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(series, color="#378ADD", linewidth=0.8)
    axes[0].set_title("Original series")
    axes[0].set_ylabel("₹/quintal")

    diff = series.diff().dropna()
    axes[1].plot(diff, color="#D85A30", linewidth=0.8)
    axes[1].set_title("First-order differenced series (d=1)")
    axes[1].set_ylabel("Δ price")
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")

    plt.suptitle("Step 2 — Stationarity: original vs differenced", fontsize=13)
    plt.tight_layout()
    save_fig("02_stationarity.png")

    return d


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — STL Decomposition
# ═══════════════════════════════════════════════════════════════════
def step3_stl_decomposition(series: pd.Series) -> int:
    print("\n" + "="*60)
    print("STEP 3: STL DECOMPOSITION")
    print("="*60)

    # Try weekly (52) and monthly (12) seasonal periods
    stl_monthly = STL(series, period=30, robust=True).fit()

    trend      = stl_monthly.trend
    seasonal   = stl_monthly.seasonal
    residual   = stl_monthly.resid

    # Seasonality strength: Fs = max(0, 1 - Var(R)/Var(S+R))
    var_resid  = np.var(residual)
    var_sr     = np.var(seasonal + residual)
    Fs         = max(0, 1 - var_resid / var_sr)
    logger.info(f"Seasonality strength (Fs): {Fs:.3f} (>0.6 = strong seasonal)")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    comps = [("Observed", series, "#378ADD"),
             ("Trend",    trend,  "#0C447C"),
             ("Seasonal", seasonal, "#1D9E75"),
             ("Residual", residual, "#EF9F27")]

    for ax, (label, data, color) in zip(axes, comps):
        ax.plot(data, color=color, linewidth=0.9)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[0].set_title(f"Step 3 — STL Decomposition | Seasonality strength Fs={Fs:.3f}")
    plt.tight_layout()
    save_fig("03_stl_decomposition.png")

    # Detect dominant seasonal period
    m = 12 if Fs > 0.4 else 1
    logger.info(f"Seasonal period m={m} selected for SARIMA")
    return m, round(Fs, 3)


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — ACF & PACF plots
# ═══════════════════════════════════════════════════════════════════
def step4_acf_pacf(series: pd.Series, d: int) -> None:
    print("\n" + "="*60)
    print("STEP 4: ACF & PACF → p, q ORDER SELECTION")
    print("="*60)

    # Work on the stationary series
    diff_series = series.diff(d).dropna() if d > 0 else series

    # Cap lags to meet statsmodels constraints:
    # - plot_acf requires nlags < len(series)
    # - plot_pacf requires nlags < 0.5 * len(series)
    # Use min(40, 0.4*len) to be conservative and leave headroom
    max_lags = min(40, max(2, int(0.4 * len(diff_series))))
    logger.info(f"Series length: {len(diff_series)}, using lags={max_lags}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(diff_series,  lags=max_lags, ax=axes[0], color="#378ADD", zero=False)
    plot_pacf(diff_series, lags=max_lags, ax=axes[1], color="#D85A30", zero=False)

    axes[0].set_title("ACF — guides MA(q) order\n(find where it cuts off inside blue band)")
    axes[1].set_title("PACF — guides AR(p) order\n(find where it cuts off inside blue band)")

    plt.suptitle("Step 4 — ACF & PACF on differenced series", fontsize=13)
    plt.tight_layout()
    save_fig("04_acf_pacf.png")

    print("\nHow to read:")
    print("  ACF:  first lag outside bands → q = that lag")
    print("  PACF: first lag outside bands → p = that lag")
    print("  Both decay slowly → try higher d or seasonal ARIMA")


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Fit ARIMA + SARIMA
# ═══════════════════════════════════════════════════════════════════
def step5_fit_models(series: pd.Series, d: int, m: int, crop: str, state: str) -> ARIMAForecaster:
    print("\n" + "="*60)
    print("STEP 5: ARIMA + SARIMA MODEL FITTING")
    print("="*60)

    results = []

    # ── 5a: Manual baseline ──────────────────────────────────────
    logger.info("\n5a. Manual ARIMA(1,d,1):")
    manual = ARIMAForecaster(crop=crop, state=state, order=(1, d, 1))
    manual.fit(series, auto=False)
    results.append({"model": f"ARIMA(1,{d},1)", "aic": manual.aic, "bic": manual.bic})

    # ── 5b: Auto-selected ARIMA ──────────────────────────────────
    logger.info("\n5b. auto_arima (non-seasonal):")
    auto_nonseasonal = ARIMAForecaster(crop=crop, state=state)
    auto_nonseasonal.auto_select_order(series, seasonal=False)
    auto_nonseasonal.fit(series, auto=False)
    results.append({"model": f"ARIMA{auto_nonseasonal.order}", "aic": auto_nonseasonal.aic, "bic": auto_nonseasonal.bic})

    # ── 5c: SARIMA (if seasonality found) ────────────────────────
    best_model = auto_nonseasonal
    # SARIMA needs enough data for seasonal patterns; require at least 4 full cycles
    min_obs_for_sarima = max(4 * m, 60)  # Both length and practical minimum
    if m > 1 and len(series) >= min_obs_for_sarima:
        try:
            logger.info(f"\n5c. auto_arima (SARIMA, m={m}):")
            sarima = ARIMAForecaster(crop=crop, state=state)
            sarima.auto_select_order(series, seasonal=True, m=m)
            sarima.fit(series, auto=False)
            results.append({"model": f"SARIMA{sarima.order}x{sarima.seasonal_order}", "aic": sarima.aic, "bic": sarima.bic})
            if sarima.aic < auto_nonseasonal.aic:
                best_model = sarima
                logger.info("SARIMA beats ARIMA on AIC — using SARIMA as best model")
        except Exception as e:
            logger.warning(f"SARIMA fit failed: {e} — using non-seasonal ARIMA")
    elif m > 1:
        logger.info(f"\n5c. SARIMA skipped: only {len(series)} observations, need ≥{min_obs_for_sarima} (4×m={m})")



    # Summary comparison table
    comp_df = pd.DataFrame(results).sort_values("aic")
    print(f"\nModel comparison:\n{comp_df.to_string(index=False)}")

    best_name = comp_df.iloc[0]["model"]
    logger.info(f"\nBest model: {best_name}")
    return best_model


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Residual Diagnostics
# ═══════════════════════════════════════════════════════════════════
def step6_residual_diagnostics(model: ARIMAForecaster) -> None:
    print("\n" + "="*60)
    print("STEP 6: RESIDUAL DIAGNOSTICS")
    print("="*60)

    diag = model.residual_diagnostics()
    print(f"\nResidual mean : {diag['residual_mean']}")
    print(f"Residual std  : {diag['residual_std']}")
    print(f"White noise   : {'YES ✓' if diag['is_white_noise'] else 'NO ✗'}")

    # statsmodels built-in diagnostic plots
    fig = model.model_fit.plot_diagnostics(figsize=(14, 8))
    plt.suptitle("Step 6 — Residual diagnostics\n(Q-Q should be straight line, correlogram inside bands)", fontsize=11)
    plt.tight_layout()
    save_fig("06_residual_diagnostics.png")


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — Forecast with Confidence Intervals
# ═══════════════════════════════════════════════════════════════════
def step7_forecast(model: ARIMAForecaster, series: pd.Series, crop: str, state: str) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 7: FORECAST — 30 AND 90 DAY")
    print("="*60)

    forecast_90 = model.forecast(steps=90)
    print(f"\nFirst 10 forecast rows:\n{forecast_90.head(10)}")

    # Plot: last 120 days of actuals + 90-day forecast
    history = series.iloc[-120:]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Actuals
    ax.plot(history.index, history.values,
            color="#378ADD", linewidth=1.2, label="Actual (last 120 days)", zorder=3)

    # Forecast
    ax.plot(forecast_90["date"], forecast_90["forecast"],
            color="#D85A30", linewidth=1.5, linestyle="--", label="Forecast", zorder=3)

    # Confidence bands
    ax.fill_between(forecast_90["date"],
                    forecast_90["lower_80"], forecast_90["upper_80"],
                    alpha=0.25, color="#D85A30", label="80% CI")
    ax.fill_between(forecast_90["date"],
                    forecast_90["lower_95"], forecast_90["upper_95"],
                    alpha=0.12, color="#D85A30", label="95% CI")

    # Divider between actual and forecast
    ax.axvline(series.index[-1], color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(series.index[-1], ax.get_ylim()[1] * 0.95, "  Forecast start",
            fontsize=9, color="gray")

    ax.set_title(f"Step 7 — {crop} @ {state}: 90-day price forecast")
    ax.set_ylabel("₹/quintal")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(loc="upper left")
    plt.tight_layout()
    save_fig("07_forecast_90day.png")

    # Save forecast CSV
    out_path = OUTPUT_DIR / "forecast_90day.csv"
    forecast_90.to_csv(out_path, index=False)
    logger.info(f"Forecast saved → {out_path}")
    return forecast_90


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — Walk-Forward Cross-Validation
# ═══════════════════════════════════════════════════════════════════
def step8_walk_forward_cv(model: ARIMAForecaster, series: pd.Series) -> dict:
    print("\n" + "="*60)
    print("STEP 8: WALK-FORWARD CROSS-VALIDATION")
    print("="*60)
    print("\nThis is the baseline score — Phase 3 (XGBoost + LSTM) must beat this MAPE\n")

    cv_results = model.walk_forward_cv(series, test_size=30, n_splits=5)

    print(f"\n{'─'*40}")
    print(f"ARIMA Baseline Results:")
    print(f"  RMSE : {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}")
    print(f"  MAE  : {cv_results['mae_mean']:.2f}")
    print(f"  MAPE : {cv_results['mape_mean']:.2f}%  ← TARGET TO BEAT IN PHASE 3")
    print(f"  Folds: {cv_results['n_folds']}")
    print(f"{'─'*40}")

    # Save baseline score for Phase 3 comparison
    baseline = pd.DataFrame([cv_results])
    baseline["model"] = f"SARIMA{model.order}x{model.seasonal_order}"
    baseline["crop"]  = model.crop
    baseline_path = OUTPUT_DIR / "baseline_scores.csv"
    baseline.to_csv(baseline_path, index=False)
    logger.info(f"Baseline scores saved → {baseline_path}")
    return cv_results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main(crop: str, state: str, auto_select_best: bool = True):
    # Try to load requested state, but if it has <200 records, auto-select better one
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
    print("\n" + "="*60)
    print("PHASE 2 — STATISTICAL MODELLING")
    print(f"Crop: {crop} | State: {state}")
    print("="*60)

    # Run all steps
    step1_raw_series(series, crop, state)
    d = step2_stationarity(series)
    m, fs = step3_stl_decomposition(series)
    step4_acf_pacf(series, d)
    best_model = step5_fit_models(series, d, m, crop, state)
    step6_residual_diagnostics(best_model)
    step7_forecast(best_model, series, crop, state)
    cv_results = step8_walk_forward_cv(best_model, series)

    # Save model artifact
    model_path = best_model.save()

    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print(f"  Model saved     : {model_path}")
    print(f"  Charts saved    : {OUTPUT_DIR}/")
    print(f"  Baseline MAPE   : {cv_results['mape_mean']}%  ← phase 3 target")
    print("="*60)


# ── Allow tuple return from step2 ─────────────────────────────────
from typing import Tuple
Tuple_d = int  # step2 returns just d for simplicity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: ARIMA statistical modeling with auto state selection"
    )
    parser.add_argument(
        "--crop", 
        default="Tomato",     
        help="Commodity name (default: Tomato)"
    )
    parser.add_argument(
        "--state", 
        default="Tamil Nadu",  
        help="State name. If state has <200 records, auto-selects best state with full history (default: Tamil Nadu)"
    )
    parser.add_argument(
        "--no-auto-select",
        action="store_true",
        help="Disable auto-state selection if specified state is too sparse"
    )
    args = parser.parse_args()
    main(args.crop, args.state, auto_select_best=not args.no_auto_select)
