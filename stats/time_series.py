"""
stats/time_series.py
--------------------
R-style time series analysis engine.

Features:
  - ADF (Augmented Dickey-Fuller) stationarity test
  - KPSS stationarity test
  - ACF / PACF computation
  - Seasonal decomposition (additive / multiplicative)
  - ARIMA / SARIMA model fitting and forecasting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.stats.time_series")


@dataclass
class TimeSeriesResult:
    test_name: str
    result: Dict[str, Any]
    series_name: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "series_name": self.series_name,
            "result": self.result,
            "warnings": self.warnings,
        }


class TimeSeriesAnalyzer:
    """
    Time series statistical analysis.

    Usage::

        tsa = TimeSeriesAnalyzer()
        adf = tsa.adf_test(df["sales"])
        decomp = tsa.seasonal_decompose(df["sales"], period=12)
        forecast = tsa.arima_forecast(df["sales"], order=(1, 1, 1), steps=12)
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def adf_test(self, series: pd.Series, regression: str = "c") -> TimeSeriesResult:
        """Augmented Dickey-Fuller test for stationarity."""
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return TimeSeriesResult(
                test_name="ADF",
                series_name=series.name or "",
                result={"error": "statsmodels not installed"},
                warnings=["statsmodels required for ADF test"],
            )
        clean = series.dropna()
        adf_stat, p, lags_used, n_obs, crit_vals, _ = adfuller(clean, regression=regression, autolag="AIC")
        is_stationary = p < self.alpha
        return TimeSeriesResult(
            test_name="ADF",
            series_name=series.name or "",
            result={
                "adf_statistic": round(float(adf_stat), 6),
                "p_value": round(float(p), 8),
                "lags_used": int(lags_used),
                "n_obs": int(n_obs),
                "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
                "is_stationary": is_stationary,
                "conclusion": "STATIONARY" if is_stationary else "NON_STATIONARY",
                "interpretation": (
                    "Series is stationary (reject unit root null)"
                    if is_stationary else
                    "Series has unit root — consider differencing"
                ),
            },
        )

    def kpss_test(self, series: pd.Series, regression: str = "c") -> TimeSeriesResult:
        """KPSS test (null = stationary, opposite to ADF)."""
        try:
            from statsmodels.tsa.stattools import kpss
        except ImportError:
            return TimeSeriesResult(
                test_name="KPSS",
                series_name=series.name or "",
                result={"error": "statsmodels not installed"},
            )
        clean = series.dropna()
        try:
            stat, p, lags, crit_vals = kpss(clean, regression=regression, nlags="auto")
        except Exception as exc:  # noqa: BLE001
            return TimeSeriesResult(test_name="KPSS", result={"error": str(exc)})
        is_stationary = p > self.alpha  # KPSS: reject H0 (stationarity) if p < alpha
        return TimeSeriesResult(
            test_name="KPSS",
            series_name=series.name or "",
            result={
                "kpss_statistic": round(float(stat), 6),
                "p_value": round(float(p), 8),
                "lags_used": int(lags),
                "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
                "is_stationary": is_stationary,
                "conclusion": "STATIONARY" if is_stationary else "NON_STATIONARY",
            },
        )

    def acf_pacf(
        self, series: pd.Series, nlags: int = 40
    ) -> Dict[str, Any]:
        """Compute ACF and PACF values."""
        try:
            from statsmodels.tsa.stattools import acf, pacf
        except ImportError:
            return {"error": "statsmodels not installed"}
        clean = series.dropna()
        actual_nlags = min(nlags, len(clean) // 2 - 1)
        acf_vals, acf_confint = acf(clean, nlags=actual_nlags, alpha=self.alpha)
        try:
            pacf_vals, pacf_confint = pacf(clean, nlags=actual_nlags, alpha=self.alpha)
        except Exception:  # noqa: BLE001
            pacf_vals = np.zeros(actual_nlags + 1)
            pacf_confint = np.zeros((actual_nlags + 1, 2))
        return {
            "series_name": series.name or "",
            "nlags": actual_nlags,
            "acf": [round(float(v), 6) for v in acf_vals],
            "acf_confidence_interval": [[round(float(ci[0]), 6), round(float(ci[1]), 6)] for ci in acf_confint],
            "pacf": [round(float(v), 6) for v in pacf_vals],
            "pacf_confidence_interval": [[round(float(ci[0]), 6), round(float(ci[1]), 6)] for ci in pacf_confint],
        }

    def seasonal_decompose(
        self, series: pd.Series, period: int = 12, model: str = "additive"
    ) -> Dict[str, Any]:
        """Classical seasonal decomposition."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose
        except ImportError:
            return {"error": "statsmodels not installed"}
        clean = series.dropna()
        if len(clean) < 2 * period:
            return {"error": f"Series too short for period={period} (need ≥ {2 * period} observations)"}
        result = sm_decompose(clean, model=model, period=period)
        return {
            "series_name": series.name or "",
            "model": model,
            "period": period,
            "trend": [round(float(v), 6) if not np.isnan(v) else None for v in result.trend],
            "seasonal": [round(float(v), 6) for v in result.seasonal],
            "residual": [round(float(v), 6) if not np.isnan(v) else None for v in result.resid],
        }

    def arima_forecast(
        self,
        series: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        steps: int = 12,
    ) -> Dict[str, Any]:
        """Fit ARIMA/SARIMA and produce a point forecast with confidence intervals."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return {"error": "statsmodels not installed"}
        clean = series.dropna()
        try:
            if seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(clean, order=order, seasonal_order=seasonal_order).fit(disp=False)
            else:
                model = ARIMA(clean, order=order).fit()
            forecast = model.get_forecast(steps=steps)
            fc_mean = forecast.predicted_mean
            ci = forecast.conf_int(alpha=self.alpha)
            return {
                "series_name": series.name or "",
                "model": "SARIMA" if seasonal_order else "ARIMA",
                "order": list(order),
                "seasonal_order": list(seasonal_order) if seasonal_order else None,
                "aic": round(float(model.aic), 4),
                "bic": round(float(model.bic), 4),
                "forecast": [round(float(v), 6) for v in fc_mean],
                "ci_lower": [round(float(v), 6) for v in ci.iloc[:, 0]],
                "ci_upper": [round(float(v), 6) for v in ci.iloc[:, 1]],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ARIMA fitting failed: {exc}"}
