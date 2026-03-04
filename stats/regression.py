"""
stats/regression.py
--------------------
R-style regression engine using statsmodels and scikit-learn.

Models:
  - OLS (Ordinary Least Squares) with full statsmodels summary
  - Logistic Regression with statsmodels GLM
  - Ridge and Lasso (sklearn) with CV coefficient paths
  - Multiple regression diagnostics: VIF, residual plots, Durbin-Watson

Each method returns a structured RegressionResult with coefficients,
p-values, confidence intervals, model fit statistics, and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.stats.regression")


@dataclass
class RegressionResult:
    model_type: str
    formula: Optional[str]
    n_obs: int
    r_squared: Optional[float]
    adj_r_squared: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    f_statistic: Optional[float]
    f_pvalue: Optional[float]
    coefficients: List[Dict[str, Any]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "formula": self.formula,
            "n_obs": self.n_obs,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "aic": self.aic,
            "bic": self.bic,
            "f_statistic": self.f_statistic,
            "f_pvalue": self.f_pvalue,
            "coefficients": self.coefficients,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
        }

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.coefficients)


class RegressionEngine:
    """
    Regression analysis engine.

    Usage::

        engine = RegressionEngine()
        result = engine.ols(df, target="price", features=["sqft", "bedrooms"])
        print(result.summary_table())
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ── OLS ──────────────────────────────────────────────────────────────────

    def ols(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        add_constant: bool = True,
    ) -> RegressionResult:
        """OLS multiple regression via statsmodels."""
        try:
            import statsmodels.api as sm
        except ImportError:
            return self._sklearn_fallback_ols(df, target, features)

        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()
        X = sub[feats]
        y = sub[target]

        if add_constant:
            X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
        except Exception as exc:  # noqa: BLE001
            return RegressionResult(
                model_type="OLS", formula=None, n_obs=len(sub),
                r_squared=None, adj_r_squared=None, aic=None, bic=None,
                f_statistic=None, f_pvalue=None, coefficients=[],
                warnings=[f"OLS fit failed: {exc}"],
            )

        coefs = []
        for name in model.params.index:
            coefs.append({
                "variable": name,
                "coefficient": round(float(model.params[name]), 6),
                "std_error": round(float(model.bse[name]), 6),
                "t_statistic": round(float(model.tvalues[name]), 4),
                "p_value": round(float(model.pvalues[name]), 6),
                "ci_lower_95": round(float(model.conf_int().loc[name, 0]), 6),
                "ci_upper_95": round(float(model.conf_int().loc[name, 1]), 6),
                "significant": bool(model.pvalues[name] < self.alpha),
            })

        # Durbin-Watson
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = float(durbin_watson(model.resid))
        except Exception:  # noqa: BLE001
            dw = None

        # VIF
        vif = self._compute_vif(sub[feats])

        return RegressionResult(
            model_type="OLS",
            formula=f"{target} ~ {' + '.join(feats)}",
            n_obs=int(model.nobs),
            r_squared=round(float(model.rsquared), 6),
            adj_r_squared=round(float(model.rsquared_adj), 6),
            aic=round(float(model.aic), 4),
            bic=round(float(model.bic), 4),
            f_statistic=round(float(model.fvalue), 4) if model.fvalue is not None else None,
            f_pvalue=round(float(model.f_pvalue), 8) if model.f_pvalue is not None else None,
            coefficients=coefs,
            diagnostics={"durbin_watson": dw, "vif": vif},
        )

    def _sklearn_fallback_ols(
        self, df: pd.DataFrame, target: str, features: Optional[List[str]]
    ) -> RegressionResult:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()
        X, y = sub[feats].values, sub[target].values
        model = LinearRegression()
        model.fit(X, y)
        r2 = float(model.score(X, y))
        coefs = [{"variable": "const", "coefficient": round(float(model.intercept_), 6)}]
        coefs += [{"variable": f, "coefficient": round(float(c), 6)} for f, c in zip(feats, model.coef_)]
        return RegressionResult(
            model_type="OLS (sklearn fallback)",
            formula=f"{target} ~ {' + '.join(feats)}",
            n_obs=len(sub), r_squared=round(r2, 6), adj_r_squared=None,
            aic=None, bic=None, f_statistic=None, f_pvalue=None,
            coefficients=coefs, warnings=["statsmodels not installed; using sklearn fallback"],
        )

    # ── Logistic ─────────────────────────────────────────────────────────────

    def logistic(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
    ) -> RegressionResult:
        """Logistic regression via statsmodels GLM (Binomial family)."""
        try:
            import statsmodels.api as sm
        except ImportError:
            return self._sklearn_logistic(df, target, features)

        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()
        X = sm.add_constant(sub[feats])
        y = sub[target]

        try:
            model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        except Exception as exc:  # noqa: BLE001
            return RegressionResult(
                model_type="Logistic", formula=None, n_obs=len(sub),
                r_squared=None, adj_r_squared=None, aic=None, bic=None,
                f_statistic=None, f_pvalue=None, coefficients=[],
                warnings=[f"Logistic fit failed: {exc}"],
            )

        coefs = []
        for name in model.params.index:
            coefs.append({
                "variable": name,
                "log_odds": round(float(model.params[name]), 6),
                "odds_ratio": round(float(np.exp(model.params[name])), 6),
                "p_value": round(float(model.pvalues[name]), 6),
                "significant": bool(model.pvalues[name] < self.alpha),
            })

        return RegressionResult(
            model_type="Logistic (GLM-Binomial)",
            formula=f"{target} ~ {' + '.join(feats)}",
            n_obs=len(sub),
            r_squared=None,
            adj_r_squared=None,
            aic=round(float(model.aic), 4),
            bic=round(float(model.bic), 4),
            f_statistic=None,
            f_pvalue=None,
            coefficients=coefs,
        )

    def _sklearn_logistic(
        self, df: pd.DataFrame, target: str, features: Optional[List[str]]
    ) -> RegressionResult:
        from sklearn.linear_model import LogisticRegression
        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()
        X, y = sub[feats].values, sub[target].values
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        coefs = [{"variable": f, "coefficient": round(float(c), 6)} for f, c in zip(feats, model.coef_[0])]
        return RegressionResult(
            model_type="Logistic (sklearn fallback)",
            formula=f"{target} ~ {' + '.join(feats)}",
            n_obs=len(sub), r_squared=None, adj_r_squared=None,
            aic=None, bic=None, f_statistic=None, f_pvalue=None,
            coefficients=coefs, warnings=["statsmodels not installed"],
        )

    # ── Ridge / Lasso ────────────────────────────────────────────────────────

    def ridge(
        self, df: pd.DataFrame, target: str, features: Optional[List[str]] = None, alpha: float = 1.0
    ) -> RegressionResult:
        from sklearn.linear_model import RidgeCV
        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()
        X, y = sub[feats].values, sub[target].values
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, alpha]).fit(X, y)
        r2 = float(model.score(X, y))
        coefs = [{"variable": f, "coefficient": round(float(c), 6)} for f, c in zip(feats, model.coef_)]
        return RegressionResult(
            model_type="Ridge (CV)",
            formula=f"{target} ~ {' + '.join(feats)}",
            n_obs=len(sub), r_squared=round(r2, 6), adj_r_squared=None,
            aic=None, bic=None, f_statistic=None, f_pvalue=None,
            coefficients=coefs,
            diagnostics={"best_alpha": float(model.alpha_)},
        )

    def lasso(
        self, df: pd.DataFrame, target: str, features: Optional[List[str]] = None
    ) -> RegressionResult:
        from sklearn.linear_model import LassoCV
        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()
        X, y = sub[feats].values, sub[target].values
        model = LassoCV(cv=5, max_iter=10000).fit(X, y)
        r2 = float(model.score(X, y))
        coefs = [{"variable": f, "coefficient": round(float(c), 6), "selected": c != 0.0}
                 for f, c in zip(feats, model.coef_)]
        return RegressionResult(
            model_type="Lasso (CV)",
            formula=f"{target} ~ {' + '.join(feats)}",
            n_obs=len(sub), r_squared=round(r2, 6), adj_r_squared=None,
            aic=None, bic=None, f_statistic=None, f_pvalue=None,
            coefficients=coefs,
            diagnostics={"best_alpha": float(model.alpha_)},
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """Compute Variance Inflation Factor for multicollinearity detection."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X_filled = X.fillna(X.mean())
            arr = X_filled.values
            return {
                col: round(float(variance_inflation_factor(arr, i)), 4)
                for i, col in enumerate(X.columns)
            }
        except Exception:  # noqa: BLE001
            return {}
