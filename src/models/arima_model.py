"""
src/models/arima_model.py
─────────────────────────
Production ARIMA / SARIMA wrapper with:
  - Manual order fitting (ARIMA & SARIMA)
  - Auto order selection via pmdarima
  - Walk-forward cross-validation
  - Forecast with confidence intervals
  - Save / load model artifacts

Usage:
    from src.models.arima_model import ARIMAForecaster

    model = ARIMAForecaster(crop="Tomato", state="Tamil Nadu")
    model.fit(series)
    forecast_df = model.forecast(steps=30)
    model.save("models/arima_tomato.pkl")
"""

import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    import pmdarima as pm
except ImportError as exc:  # pragma: no cover - dependency issue
    pm = None  # type: ignore
    _PMDARIMA_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class ARIMAForecaster:
    """
    ARIMA / SARIMA forecaster for a single crop-state time series.

    Parameters:
        crop   : commodity name (for logging/saving)
        state  : state name (for logging/saving)
        order  : (p, d, q) — set None to use auto_arima
        seasonal_order : (P, D, Q, m) — set None for non-seasonal
    """

    def __init__(
        self,
        crop:           str = "unknown",
        state:          str = "unknown",
        order:          Optional[Tuple] = None,
        seasonal_order: Optional[Tuple] = None,
    ):
        self.crop           = crop
        self.state          = state
        self.order          = order
        self.seasonal_order = seasonal_order
        self.model_fit      = None
        self.series         = None
        self.aic            = None
        self.bic            = None

    # ──────────────────────────────────────────────────────────────────────
    # 1. Stationarity Tests
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def adf_test(series: pd.Series, verbose: bool = True) -> dict:
        """
        Augmented Dickey-Fuller test.
        H0: series has unit root (non-stationary)
        Reject H0 (p < 0.05) → stationary
        """
        result = adfuller(series.dropna(), autolag="AIC")
        output = {
            "test_statistic": round(result[0], 4),
            "p_value":        round(result[1], 4),
            "n_lags":         result[2],
            "n_obs":          result[3],
            "is_stationary":  result[1] < 0.05,
            "critical_values": result[4],
        }
        if verbose:
            status = "STATIONARY ✓" if output["is_stationary"] else "NON-STATIONARY ✗"
            logger.info(f"ADF Test — {status} | p={output['p_value']} | stat={output['test_statistic']}")
        return output

    @staticmethod
    def kpss_test(series: pd.Series, verbose: bool = True) -> dict:
        """
        KPSS test.
        H0: series is stationary (level or trend stationary)
        Reject H0 (p < 0.05) → non-stationary
        """
        result = kpss(series.dropna(), regression="c", nlags="auto")
        output = {
            "test_statistic": round(result[0], 4),
            "p_value":        round(result[1], 4),
            "is_stationary":  result[1] >= 0.05,
            "critical_values": result[3],
        }
        if verbose:
            status = "STATIONARY ✓" if output["is_stationary"] else "NON-STATIONARY ✗"
            logger.info(f"KPSS Test — {status} | p={output['p_value']} | stat={output['test_statistic']}")
        return output

    def check_stationarity(self, series: pd.Series) -> Tuple[bool, int]:
        """
        Run ADF + KPSS. Return (is_stationary, recommended_d).
        Agreement: both say stationary → d=0
        ADF says non-stationary → d=1, then recheck
        """
        logger.info("─── Stationarity Check ───────────────────")
        adf  = self.adf_test(series)
        kpss_ = self.kpss_test(series)

        if adf["is_stationary"] and kpss_["is_stationary"]:
            logger.info("Both tests agree: series is STATIONARY → d=0")
            return True, 0
        elif not adf["is_stationary"] and not kpss_["is_stationary"]:
            logger.info("Both tests agree: series is NON-STATIONARY → try d=1")
            diff = series.diff().dropna()
            adf2 = self.adf_test(diff, verbose=False)
            d = 1 if adf2["is_stationary"] else 2
            logger.info(f"After {d}x differencing → stationary")
            return False, d
        else:
            logger.info("Tests disagree — defaulting to d=1 (conservative)")
            return False, 1

    # ──────────────────────────────────────────────────────────────────────
    # 2. Auto Order Selection
    # ──────────────────────────────────────────────────────────────────────
    def auto_select_order(self, series: pd.Series, seasonal: bool = True, m: int = 12) -> None:
        """
        Use pmdarima auto_arima to find best (p,d,q)(P,D,Q,m) by AIC.
        Sets self.order and self.seasonal_order.
        """
        if pm is None:  # pragma: no cover - dependency issue
            raise ImportError(
                "pmdarima is not installed. Install requirements.txt to use auto_arima."
            ) from _PMDARIMA_IMPORT_ERROR
        logger.info("Running auto_arima — this may take a minute...")
        auto_model = pm.auto_arima(
            series,
            seasonal=seasonal,
            m=m,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        self.order          = auto_model.order
        self.seasonal_order = auto_model.seasonal_order if seasonal else None
        logger.info(f"auto_arima selected: ARIMA{self.order} x {self.seasonal_order}")
        logger.info(f"AIC: {auto_model.aic():.2f}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Fit
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, series: pd.Series, auto: bool = False, m: int = 12) -> "ARIMAForecaster":
        """
        Fit SARIMA model to price series.

        Args:
            series : pd.Series with DatetimeIndex, values = modal_price
            auto   : use auto_arima to find order if self.order is None
            m      : seasonal period (12 = monthly, 52 = weekly)
        """
        self.series = series.copy().asfreq("D").fillna(method="ffill")

        if self.order is None or auto:
            self.auto_select_order(self.series, seasonal=True, m=m)

        logger.info(f"Fitting SARIMA{self.order}x{self.seasonal_order} on {len(series)} obs")

        model = SARIMAX(
            self.series,
            order=self.order,
            seasonal_order=self.seasonal_order or (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)
        self.aic = round(self.model_fit.aic, 2)
        self.bic = round(self.model_fit.bic, 2)
        logger.info(f"Model fitted — AIC: {self.aic} | BIC: {self.bic}")
        return self

    # ──────────────────────────────────────────────────────────────────────
    # 4. Forecast
    # ──────────────────────────────────────────────────────────────────────
    def forecast(self, steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generate forecast with confidence intervals.

        Returns DataFrame columns:
            date, forecast, lower_80, upper_80, lower_95, upper_95
        """
        if self.model_fit is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")

        pred = self.model_fit.get_forecast(steps=steps)
        mean = pred.predicted_mean
        ci_95 = pred.conf_int(alpha=0.05)
        ci_80 = pred.conf_int(alpha=0.20)

        df = pd.DataFrame({
            "date":     mean.index,
            "forecast": mean.values.round(2),
            "lower_80": ci_80.iloc[:, 0].values.round(2),
            "upper_80": ci_80.iloc[:, 1].values.round(2),
            "lower_95": ci_95.iloc[:, 0].values.round(2),
            "upper_95": ci_95.iloc[:, 1].values.round(2),
        })
        df["crop"]   = self.crop
        df["state"]  = self.state
        df["model"]  = f"SARIMA{self.order}x{self.seasonal_order}"
        return df

    # ──────────────────────────────────────────────────────────────────────
    # 5. Residual Diagnostics
    # ──────────────────────────────────────────────────────────────────────
    def residual_diagnostics(self) -> dict:
        """
        Run Ljung-Box test on residuals to check for autocorrelation.
        Good model residuals = white noise (p > 0.05 at each lag)
        """
        if self.model_fit is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        residuals = self.model_fit.resid
        lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        is_white_noise = (lb_result["lb_pvalue"] > 0.05).all()

        stats = {
            "residual_mean":   round(float(residuals.mean()), 4),
            "residual_std":    round(float(residuals.std()), 4),
            "is_white_noise":  bool(is_white_noise),
            "ljung_box":       lb_result.round(4).to_dict(),
        }
        status = "PASS ✓" if is_white_noise else "FAIL ✗ — model may be mis-specified"
        logger.info(f"Residual diagnostics — Ljung-Box: {status}")
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # 6. Walk-Forward Cross-Validation
    # ──────────────────────────────────────────────────────────────────────
    def walk_forward_cv(
        self,
        series:     pd.Series,
        test_size:  int = 30,
        n_splits:   int = 5,
    ) -> dict:
        """
        Walk-forward (rolling origin) cross-validation.
        DO NOT use random train/test split on time series — it leaks.

        Returns dict with RMSE, MAE, MAPE across all folds.
        """
        logger.info(f"Walk-forward CV: {n_splits} folds, {test_size}-day test window each")
        total    = len(series)
        min_train = total - (n_splits * test_size)

        rmse_list, mae_list, mape_list = [], [], []

        for i in range(n_splits):
            train_end = min_train + (i * test_size)
            train     = series.iloc[:train_end]
            test      = series.iloc[train_end: train_end + test_size]

            try:
                fold_model = SARIMAX(
                    train,
                    order=self.order or (1, 1, 1),
                    seasonal_order=self.seasonal_order or (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fold_fit  = fold_model.fit(disp=False)
                pred      = fold_fit.forecast(steps=len(test))
                actual    = test.values
                predicted = pred.values

                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                mae  = np.mean(np.abs(actual - predicted))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100

                rmse_list.append(rmse)
                mae_list.append(mae)
                mape_list.append(mape)
                logger.info(f"  Fold {i+1}/{n_splits} — RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

            except Exception as e:
                logger.warning(f"  Fold {i+1} failed: {e}")

        results = {
            "rmse_mean": round(float(np.mean(rmse_list)), 2),
            "mae_mean":  round(float(np.mean(mae_list)),  2),
            "mape_mean": round(float(np.mean(mape_list)), 2),
            "rmse_std":  round(float(np.std(rmse_list)),  2),
            "n_folds":   len(rmse_list),
        }
        logger.info(f"\nCV Results → RMSE: {results['rmse_mean']} | MAE: {results['mae_mean']} | MAPE: {results['mape_mean']}%")
        return results

    # ──────────────────────────────────────────────────────────────────────
    # 7. Save / Load
    # ──────────────────────────────────────────────────────────────────────
    def save(self, path: str = None) -> str:
        tag  = f"{self.crop}_{self.state}".lower().replace(" ", "_")
        path = path or str(MODELS_DIR / f"arima_{tag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ARIMAForecaster":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model

    def summary(self) -> str:
        if self.model_fit is None:
            return "Model not fitted."
        return str(self.model_fit.summary())
