"""
preprocessing/analyst_brain.py
================================
The Analyst Intelligence Brain — DIPEX Stage 0.4

This module gives the pipeline the reasoning capability of a senior expert
data analyst. Every decision is:
  • Context-aware  — based on what the column actually represents
  • Data-driven    — based on statistics measured from the real data
  • Transparent    — every decision is logged with a plain-English reason
  • Conservative   — when uncertain, the safer option is always chosen

The Brain runs BEFORE Triage so that downstream stages inherit its decisions
(transformation strategy, outlier policy, imputation hint, semantic label).

Design Principles
-----------------
  1. No silent decisions. Every action has a reason logged.
  2. No irreversible destructive operations. Flag; never hard-delete.
  3. Domain reasoning first. Statistics second. Default last.
  4. If unsure → log uncertainty → apply the safest fallback.
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Semantic Type labels ───────────────────────────────────────────────────────
class SemanticType:
    ID          = "id"            # unique identifier  — drop before modeling
    DATETIME    = "datetime"      # date / timestamp  — extract features
    EMAIL       = "email"         # contact info      — drop / anonymise
    PHONE       = "phone"         # contact info      — drop / anonymise
    URL         = "url"           # web link          — drop / hash
    CURRENCY    = "currency"      # financial value   — treat as numeric
    PERCENTAGE  = "percentage"    # 0-100 or 0-1 range
    BINARY      = "binary"        # exactly 2 unique values
    CATEGORICAL = "categorical"   # low-cardinality string
    TEXT        = "free_text"     # high-cardinality string — embed / drop
    NUMERIC     = "numeric"       # continuous numeric
    COUNT       = "count"         # non-negative integer
    BOOLEAN     = "boolean"       # True/False column
    CONSTANT    = "constant"      # zero variance — useless
    UNKNOWN     = "unknown"


# ── Per-column decision ────────────────────────────────────────────────────────
@dataclass
class ColumnDecision:
    col: str
    semantic_type: str = SemanticType.UNKNOWN

    # Recommended preprocessing actions
    transform_strategy: str = "none"      # log1p | sqrt | yeo-johnson | none
    outlier_strategy: str   = "iqr"       # iqr | winsorise | flag | none
    imputation_hint: str    = "median"    # median | mode | knn | mice | none | forward_fill
    should_drop: bool       = False
    drop_reason: str        = ""

    # Business-rule violations found
    violations: List[str] = field(default_factory=list)

    # Natural-language analyst reasoning
    reasoning: List[str] = field(default_factory=list)

    # Raw stats (filled by brain)
    null_rate: float        = 0.0
    unique_rate: float      = 0.0
    skewness: float         = 0.0
    n_unique: int           = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "col": self.col,
            "semantic_type": self.semantic_type,
            "transform_strategy": self.transform_strategy,
            "outlier_strategy": self.outlier_strategy,
            "imputation_hint": self.imputation_hint,
            "should_drop": self.should_drop,
            "drop_reason": self.drop_reason,
            "violations": self.violations,
            "reasoning": self.reasoning,
            "null_rate": round(float(self.null_rate), 4),
            "unique_rate": round(float(self.unique_rate), 4),
            "skewness": round(float(self.skewness), 4),
            "n_unique": self.n_unique,
        }


# ── Full brain report ──────────────────────────────────────────────────────────
@dataclass
class BrainReport:
    run_id: str
    original_shape: Tuple[int, int] = (0, 0)
    column_decisions: Dict[str, ColumnDecision] = field(default_factory=dict)
    dataset_level_notes: List[str] = field(default_factory=list)
    recommended_target: Optional[str] = None
    detected_domain: str = "general"   # finance | medical | hr | ecommerce | general
    data_health_score: float = 0.0     # 0–100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_shape": self.original_shape,
            "detected_domain": self.detected_domain,
            "data_health_score": round(float(self.data_health_score), 2),
            "recommended_target": self.recommended_target,
            "dataset_level_notes": self.dataset_level_notes,
            "column_decisions": {
                col: dec.to_dict()
                for col, dec in self.column_decisions.items()
            },
        }


# ── The Brain ─────────────────────────────────────────────────────────────────
class AnalystBrain:
    """
    Senior Expert Data Analyst Intelligence Engine.

    Runs before Triage to annotate every column with:
      - What it semantically represents
      - The correct preprocessing strategy for it
      - Business rule violations detected
      - A plain-English explanation of every decision

    Returns:
      annotated_df   — df with metadata attached via attrs
      brain_report   — full ColumnDecision map
    """

    # Sentinel patterns a real analyst knows about
    _STR_SENTINELS = frozenset({
        "n/a", "na", "nan", "none", "null", "nil", "missing", "unknown",
        "not available", "not applicable", "-", "--", "---", "?", ".",
        "undefined", "n.a.", "n.a", "#n/a", "#null!", "blank", "empty",
        "tbd", "tbc", "#value!", "#ref!", "#name?",
    })
    _NUM_SENTINELS = frozenset({-999, -9999, -1, 9999, 99999, -99, 999})

    # Domain keyword maps
    _DOMAIN_SIGNALS = {
        "finance":   ["salary", "revenue", "profit", "loss", "income", "tax", "loan",
                      "credit", "debit", "account", "balance", "expense", "cost",
                      "price", "amount", "payment", "transaction", "fund", "equity"],
        "medical":   ["age", "diagnosis", "patient", "drug", "dose", "treatment",
                      "hospital", "disease", "symptom", "clinical", "bmi", "blood",
                      "glucose", "cholesterol", "mortality", "survival"],
        "hr":        ["employee", "department", "hire", "tenure", "attrition",
                      "performance", "rating", "manager", "team", "role"],
        "ecommerce": ["product", "order", "customer", "purchase", "cart", "sku",
                      "review", "rating", "shipping", "return", "refund"],
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._cfg = config.get("analyst_brain", {}) if config else {}

    # ── Public API ─────────────────────────────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        run_id: str = "unknown",
        target_col: Optional[str] = None,
        rl_recommendations: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, BrainReport]:
        """
        Analyse the dataset with senior analyst reasoning.
        Returns the (potentially lightly corrected) df and a full BrainReport.
        """
        report = BrainReport(run_id=run_id, original_shape=df.shape)

        if df.empty or len(df.columns) == 0:
            report.dataset_level_notes.append("Dataset is empty — no analysis possible.")
            return df, report

        df = df.copy()

        # Step 1 — Dataset-level domain detection
        report.detected_domain   = self._detect_domain(df)
        report.dataset_level_notes.append(
            f"Detected domain: {report.detected_domain.upper()}. "
            f"Rules and heuristics will be tuned accordingly."
        )

        # Step 2 — Replace known string sentinels with NaN
        df = self._replace_string_sentinels(df, report)

        # Step 3 — Per-column analysis
        for col in df.columns:
            decision = self._analyse_column(df, col, target_col, report.detected_domain, rl_recommendations)
            report.column_decisions[col] = decision
            self._apply_safe_fixes(df, col, decision)

        # Step 4 — Cross-column reasoning
        self._cross_column_reasoning(df, target_col, report)

        # Step 5 — Auto-detect recommended target if none given
        if target_col is None:
            report.recommended_target = self._suggest_target(df, report)
            if report.recommended_target:
                report.dataset_level_notes.append(
                    f"No target column specified. Best candidate: "
                    f"'{report.recommended_target}' (binary/low-cardinality column detected)."
                )

        # Step 6 — Compute data health score
        report.data_health_score = self._score_health(df, report)
        report.dataset_level_notes.append(
            f"Data Health Score: {report.data_health_score:.1f}/100"
        )

        # Step 7 — Attach brain report to df.attrs for downstream stages
        df.attrs["brain_report"] = report.to_dict()
        df.attrs["column_decisions"] = {
            col: dec.to_dict() for col, dec in report.column_decisions.items()
        }

        logger.info(
            "[%s] AnalystBrain: domain=%s health=%.1f cols=%d "
            "drops_recommended=%d violations=%d",
            str(run_id)[:8],
            report.detected_domain,
            report.data_health_score,
            len(df.columns),
            sum(1 for d in report.column_decisions.values() if d.should_drop),
            sum(len(d.violations) for d in report.column_decisions.values()),
        )
        return df, report

    # ── Domain Detection ───────────────────────────────────────────────────────
    def _detect_domain(self, df: pd.DataFrame) -> str:
        col_str = " ".join(df.columns.str.lower().tolist())
        
        # 1. Try ML Domain Classifier
        try:
            import joblib
            import os
            # Assume models are stored in dipex_project/models/ or similar 
            # (or use config if available, but let's just attempt a relative/absolute path)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "domain_classifier.pkl")
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                proba = clf.predict_proba([col_str])[0]
                best_idx = np.argmax(proba)
                # Only use ML if confident
                if proba[best_idx] > 0.4:
                    return str(clf.classes_[best_idx])
        except Exception as e:
            logger.debug(f"[AnalystBrain] ML Domain Classifier failed: {e}. Falling back to heuristics.")

        # 2. Heuristic Fallback
        scores: Dict[str, int] = {}
        for domain, keywords in self._DOMAIN_SIGNALS.items():
            scores[domain] = sum(1 for kw in keywords if kw in col_str)
        best = max(scores.keys(), key=lambda k: scores[k])
        return str(best) if scores[best] >= 2 else "general"

    # ── Sentinel Replacement ──────────────────────────────────────────────────
    def _replace_string_sentinels(self, df: pd.DataFrame, report: BrainReport) -> pd.DataFrame:
        total = 0
        for col in df.select_dtypes(include="object").columns:
            mask = df[col].astype(str).str.strip().str.lower().isin(self._STR_SENTINELS)
            n = int(mask.sum())
            if n > 0:
                df.loc[mask, col] = np.nan
                total = int(total + n)
        if total > 0:
            report.dataset_level_notes.append(
                f"Replaced {total} string sentinel values "
                f"(e.g. 'N/A', 'null', '--') with NaN across all columns."
            )
        return df

    # ── Column Analysis ───────────────────────────────────────────────────────
    def _analyse_column(
        self,
        df: pd.DataFrame,
        col: str,
        target_col: Optional[str],
        domain: str,
        rl_recommendations: Optional[Dict[str, Any]] = None,
    ) -> ColumnDecision:
        dec = ColumnDecision(col=col)
        s   = df[col]
        n   = len(s)

        # Basic stats
        dec.null_rate   = float(s.isna().mean())
        try:
            dec.n_unique    = int(s.dropna().nunique())
        except Exception: # noqa: BLE001
            dec.n_unique    = int(s.dropna().astype(str).nunique())
        dec.unique_rate = dec.n_unique / max(n, 1)

        # ── A. Semantic type ─────────────────────────────────────────────────
        dec.semantic_type = self._infer_semantic_type(s, col, dec)

        # ── B. Should this column be dropped? ───────────────────────────────
        self._assess_drop(dec, col, target_col)

        if not dec.should_drop:
            # ── C. Distribution & transform strategy (numeric only) ───────────
            if dec.semantic_type in (SemanticType.NUMERIC, SemanticType.CURRENCY,
                                      SemanticType.PERCENTAGE, SemanticType.COUNT):
                self._assess_transform(s, dec, domain)

            # ── D. Outlier strategy (Consult RL) ──────────────────────────
            if dec.semantic_type in (SemanticType.NUMERIC, SemanticType.CURRENCY,
                                      SemanticType.COUNT):
                self._assess_outlier_strategy(s, dec, rl_recommendations)

            # ── E. Imputation hint (Consult RL) ───────────────────────────
            self._assess_imputation(s, dec, rl_recommendations)

        # ── F. Business rule violations ──────────────────────────────────
        self._check_business_rules(s, col, dec, domain)

        # ── G. Final reasoning summary ───────────────────────────────────
        dec.reasoning.append(
            f"Semantic type: {dec.semantic_type} | "
            f"Null rate: {dec.null_rate:.1%} | "
            f"Unique values: {dec.n_unique} ({dec.unique_rate:.1%} of rows)"
        )
        return dec

    # ── Semantic Type Inference ────────────────────────────────────────────────
    def _infer_semantic_type(self, s: pd.Series, col: str, dec: ColumnDecision) -> str:
        col_l = col.lower()
        non_null = s.dropna()

        # Constant — zero variance
        if dec.n_unique <= 1:
            dec.reasoning.append("Column has ≤1 unique value — it is constant and carries no information.")
            return SemanticType.CONSTANT

        # Boolean
        if set(non_null.unique()) <= {True, False, 0, 1, "0", "1", "yes", "no",
                                       "true", "false", "y", "n"}:
            dec.reasoning.append("Column contains only boolean-like values (True/False, Y/N, 0/1).")
            return SemanticType.BOOLEAN

        # ID detection — high uniqueness + name patterns
        id_patterns = ["_id", "id_", "^id$", "uuid", "guid", "code", "key", "ref", "index", "idx"]
        is_id_name  = any(p in col_l for p in id_patterns)
        if dec.unique_rate > 0.90 and dec.n_unique > 50:
            dec.reasoning.append(
                f"Column is {dec.unique_rate:.0%} unique — virtually every row has a different value. "
                "This is an identifier column. It adds no predictive value and will be flagged for removal."
            )
            return SemanticType.ID
        if is_id_name and dec.unique_rate > 0.50:
            dec.reasoning.append("Column name suggests an identifier and has high cardinality — treated as ID.")
            return SemanticType.ID

        # Datetime — try parsing sample
        if s.dtype == "object" or str(s.dtype).startswith("datetime"):
            sample = non_null.head(20).astype(str)
            parsed_count = sum(1 for v in sample if self._looks_like_date(v))
            if parsed_count >= len(sample) * 0.7:
                dec.reasoning.append(
                    "Column values look like dates/timestamps. "
                    "Will be parsed and decomposed into year, month, day, weekday features."
                )
                return SemanticType.DATETIME

        # Email
        if s.dtype == "object":
            email_mask = non_null.astype(str).str.match(r"^[\w.+-]+@[\w-]+\.\w+$", na=False)
            if email_mask.mean() > 0.6:
                dec.reasoning.append("Column contains email addresses — PII. Will be flagged for governance.")
                return SemanticType.EMAIL

            # Phone
            phone_mask = non_null.astype(str).str.replace(r"[\s\-\(\)\+]", "", regex=True).str.match(
                r"^\d{7,15}$", na=False
            )
            if phone_mask.mean() > 0.6:
                dec.reasoning.append("Column contains phone numbers — PII. Will be flagged for governance.")
                return SemanticType.PHONE

            # URL
            url_mask = non_null.astype(str).str.match(r"^https?://", na=False)
            if url_mask.mean() > 0.5:
                dec.reasoning.append("Column contains URLs — no direct analytical value.")
                return SemanticType.URL

        # Currency — detect $ £ € symbols or column name hints
        currency_name  = any(p in col_l for p in ["price", "cost", "salary", "revenue",
                                                    "income", "fee", "amount", "payment",
                                                    "spend", "budget", "wage", "earning"])
        currency_val   = (
            s.dtype == "object" and
            non_null.astype(str).str.contains(r"[$£€¥]", regex=True).mean() > 0.3
        )
        if currency_name or currency_val:
            dec.reasoning.append(
                "Column represents a monetary value. "
                "Will be cleaned of currency symbols and treated as numeric."
            )
            return SemanticType.CURRENCY

        # Percentage
        pct_name = any(p in col_l for p in ["percent", "pct", "rate", "ratio", "share", "fraction"])
        if pct_name:
            dec.reasoning.append("Column appears to be a ratio or percentage.")
            return SemanticType.PERCENTAGE
        if pd.api.types.is_numeric_dtype(s):
            vals = non_null
            if vals.min() >= 0 and vals.max() <= 1 and pct_name is False:
                if not any(p in col_l for p in ["flag", "binary", "is_", "has_"]):
                    dec.reasoning.append(
                        "Values are in [0, 1] range — could be a proportion or probability."
                    )
                    return SemanticType.PERCENTAGE

        # Count data — non-negative integers
        if pd.api.types.is_numeric_dtype(s):
            vals = non_null
            if vals.min() >= 0 and (vals.astype(float) == vals.astype(int).astype(float)).all():
                count_names = ["count", "qty", "quantity", "num_", "n_", "visits", "sessions",
                               "orders", "purchases", "clicks", "views", "days", "months"]
                if any(p in col_l for p in count_names) or vals.max() < 10000:
                    dec.reasoning.append(
                        "Column appears to be count data (non-negative integers). "
                        "Will apply sqrt transform to reduce skew."
                    )
                    return SemanticType.COUNT

        # Numeric
        if pd.api.types.is_numeric_dtype(s):
            dec.reasoning.append("Standard numeric column.")
            return SemanticType.NUMERIC

        # ── B5 fix: Numeric string detection ────────────────────────────────
        # Object columns that are actually numeric strings (e.g. '1.5', '2,000.50')
        # would fall through to CATEGORICAL without this check.
        if s.dtype == "object" and len(non_null) >= 3:
            # Strip common numeric formatting (currency symbols, commas, whitespace)
            cleaned = non_null.astype(str).str.replace(r"[$£€¥,\s]", "", regex=True).str.strip()
            # Attempt coercion — a column is "numeric string" if ≥80% parse cleanly
            numeric_coerced = pd.to_numeric(cleaned, errors="coerce")
            numeric_frac = numeric_coerced.notna().sum() / max(len(numeric_coerced), 1)
            if numeric_frac >= 0.80:
                dec.reasoning.append(
                    f"Column has object dtype but {numeric_frac:.0%} of values parse as numbers "
                    "(numeric strings detected). Reclassifying as NUMERIC and scheduling coercion."
                )
                # Check for currency formatting (symbols present before stripping)
                has_currency_sym = (
                    non_null.astype(str).str.contains(r"[$£€¥]", regex=True).mean() > 0.1
                )
                if has_currency_sym or currency_name:
                    return SemanticType.CURRENCY
                return SemanticType.NUMERIC

        # Binary categorical
        if dec.n_unique == 2:
            unique_vals = list(non_null.unique())
            dec.reasoning.append(f"Exactly 2 unique values — binary column ({unique_vals[:2]}).")
            return SemanticType.BINARY

        # High-cardinality string — free text
        if s.dtype == "object" and dec.unique_rate > 0.5:
            dec.reasoning.append(
                f"High cardinality string ({dec.unique_rate:.0%} unique) — likely free text. "
                "No analytical value unless embedded. Will be flagged for removal."
            )
            return SemanticType.TEXT

        # Low-cardinality string — categorical
        if s.dtype == "object":
            dec.reasoning.append(
                f"Low-cardinality string with {dec.n_unique} unique values — categorical."
            )
            return SemanticType.CATEGORICAL

        return SemanticType.UNKNOWN

    # ── Drop Assessment ───────────────────────────────────────────────────────
    def _assess_drop(self, dec: ColumnDecision, col: str, target_col: Optional[str]) -> None:
        if col == target_col:
            return  # Never drop the target

        drop_types = {
            SemanticType.CONSTANT: "Zero variance — no information content. A data analyst would remove this immediately.",
            SemanticType.ID: "Identifier column — unique per row, no predictive value. Remove before modeling.",
            SemanticType.EMAIL: "Email is PII and has no predictive value for modeling. Flag for governance.",
            SemanticType.PHONE: "Phone number is PII. Flag for governance and remove from feature set.",
            SemanticType.URL: "URL string has no analytical value in raw form.",
            SemanticType.TEXT: "Free text is too high-cardinality for direct use without NLP embedding.",
        }
        if dec.semantic_type in drop_types:
            dec.should_drop = True
            dec.drop_reason = drop_types[dec.semantic_type]
            dec.reasoning.append(f"⚠ DROP RECOMMENDED: {dec.drop_reason}")

        if dec.null_rate >= 0.95:
            dec.should_drop   = True
            dec.drop_reason   = f"Column is {dec.null_rate:.1%} null — virtually no usable data remains."
            dec.reasoning.append(f"⚠ DROP RECOMMENDED: {dec.drop_reason}")

    # ── Transform Strategy ────────────────────────────────────────────────────
    def _assess_transform(
        self, s: pd.Series, dec: ColumnDecision, domain: str
    ) -> None:
        non_null = s.dropna()
        if non_null.empty:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dec.skewness = float(non_null.skew())
        except Exception:
            dec.skewness = 0.0

        abs_skew = abs(dec.skewness)

        if dec.semantic_type == SemanticType.COUNT:
            dec.transform_strategy = "sqrt"
            dec.reasoning.append(
                f"Count data — applying √ (sqrt) transform. "
                f"Skewness was {dec.skewness:.2f}."
            )
        elif non_null.min() >= 0 and abs_skew > 1.0:
            dec.transform_strategy = "log1p"
            dec.reasoning.append(
                f"Right-skewed positive data (skewness={dec.skewness:.2f} > 1.0). "
                "Applying log(1+x) transform — standard analyst practice for income, price, revenue."
            )
        elif abs_skew > 0.5:
            dec.transform_strategy = "yeo-johnson"
            dec.reasoning.append(
                f"Moderately skewed (skewness={dec.skewness:.2f}). "
                "Applying Yeo-Johnson transform — works for positive and negative values."
            )
        else:
            dec.transform_strategy = "none"
            dec.reasoning.append(
                f"Distribution is approximately symmetric (skewness={dec.skewness:.2f}). "
                "No transform needed — apply standard scaling."
            )

    # ── Outlier Strategy ──────────────────────────────────────────────────────
    def _assess_outlier_strategy(self, s: pd.Series, dec: ColumnDecision,
                                 rl_recs: Optional[Dict[str, Any]] = None) -> None:
        non_null = s.dropna()
        if non_null.empty or not pd.api.types.is_numeric_dtype(non_null):
            return
            
        # RL Influence
        rl_policy = rl_recs.get("outlier_policy", {}).get("recommended") if rl_recs else None
        
        try:
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr    = q3 - q1
            fence  = 1.5 * iqr
            n_out  = int(((non_null < q1 - fence) | (non_null > q3 + fence)).sum())
            pct    = n_out / max(len(non_null), 1)

            if rl_policy == "aggressive_winsorize" or pct > 0.15:
                dec.outlier_strategy = "winsorise"
                reason = "Heavy-tailed distribution" if pct > 0.15 else "RL preference: aggressive_winsorize"
                dec.reasoning.append(
                    f"{pct:.1%} of values fall beyond 1.5×IQR. Strategy: {dec.outlier_strategy} ({reason})."
                )
            elif rl_policy == "preserve_signal":
                dec.outlier_strategy = "flag"
                dec.reasoning.append("RL preference: preserve_signal. Flagging outliers rather than clipping.")
            elif pct > 0.01:
                dec.outlier_strategy = "iqr"
                dec.reasoning.append(f"Standard IQR capping applied ({n_out} outliers).")
            else:
                dec.outlier_strategy = "flag"
                dec.reasoning.append(f"Low outlier count ({pct:.1%}). Flagging only.")
        except Exception:
            dec.outlier_strategy = "iqr"

    # ── Imputation Hint ───────────────────────────────────────────────────────
    def _assess_imputation(self, s: pd.Series, dec: ColumnDecision,
                             rl_recs: Optional[Dict[str, Any]] = None) -> None:
        if dec.null_rate == 0 or dec.should_drop:
            dec.imputation_hint = "none"
            return

        rl_policy = rl_recs.get("imputation_preference", {}).get("recommended") if rl_recs else None

        # Temporal columns → forward fill (last known value)
        if dec.semantic_type == SemanticType.DATETIME:
            dec.imputation_hint = "forward_fill"
            return

        # Categorical / binary → mode
        if dec.semantic_type in (SemanticType.CATEGORICAL, SemanticType.BINARY, SemanticType.BOOLEAN):
            dec.imputation_hint = "mode"
            return

        # RL Influence on imputation hint
        if rl_policy == "robust_fast (Iterative Median)":
            dec.imputation_hint = "median"
            dec.reasoning.append("RL preference: robust_fast. Choosing median over slower statistical methods.")
            return
        elif rl_policy == "distribution_preserving (SMOTE-assisted)":
             # Brain interprets this as a preference for high-quality MICE with indicators
             if dec.null_rate > 0.05:
                 dec.imputation_hint = "mice"
                 dec.reasoning.append("RL preference: distribution_preserving. Promoting to MICE.")
                 return

        # Heuristics if no strong RL signal
        if dec.null_rate > 0.30:
            dec.imputation_hint = "mice"
        elif dec.null_rate > 0.10:
            dec.imputation_hint = "knn"
        else:
            dec.imputation_hint = "median"

    # ── Business Rule Validation ──────────────────────────────────────────────
    def _check_business_rules(
        self,
        s: pd.Series,
        col: str,
        dec: ColumnDecision,
        domain: str,
    ) -> None:
        col_l = col.lower()
        non_null = s.dropna()
        if non_null.empty:
            return

        # --- Positive-only fields ---
        positive_only = ["age", "salary", "income", "price", "cost", "quantity",
                         "qty", "amount", "revenue", "bmi", "weight", "height",
                         "duration", "tenure", "years", "count", "visits"]
        if any(p in col_l for p in positive_only) and pd.api.types.is_numeric_dtype(s):
            n_neg = int((non_null < 0).sum())
            if n_neg > 0:
                dec.violations.append(
                    f"{n_neg} negative values in '{col}' — "
                    f"this field should always be ≥ 0 (e.g. age, salary, quantity cannot be negative)."
                )
                dec.reasoning.append(
                    f"Business rule violation: {n_neg} negative values in a field "
                    f"that should be strictly positive. Will flag for correction."
                )

        # --- Age bounds ---
        if "age" in col_l and pd.api.types.is_numeric_dtype(s):
            if non_null.max() > 150:
                dec.violations.append(
                    f"Age column has values > 150 ({non_null.max():.0f}) — clearly impossible."
                )
            if non_null.min() < 0:
                dec.violations.append("Age column has negative values.")

        # --- Percentage bounds ---
        if dec.semantic_type == SemanticType.PERCENTAGE and pd.api.types.is_numeric_dtype(s):
            expected_max = 100 if non_null.max() > 1 else 1
            if non_null.min() < 0:
                dec.violations.append(f"Percentage column '{col}' has values < 0.")
            if non_null.max() > expected_max * 1.01:
                dec.violations.append(
                    f"Percentage column '{col}' has values > {expected_max} "
                    f"(max={non_null.max():.2f}) — out of valid range."
                )

        # --- Date in the future (for birth dates, hire dates) ---
        birth_hire = ["birth", "dob", "born", "hire", "start", "join"]
        if any(p in col_l for p in birth_hire) and dec.semantic_type == SemanticType.DATETIME:
            try:
                parsed = pd.to_datetime(non_null, errors="coerce").dropna()
                n_future = int((parsed > pd.Timestamp.now()).sum())
                if n_future > 0:
                    dec.violations.append(
                        f"{n_future} future dates in '{col}' — "
                        "birth/hire dates cannot be in the future."
                    )
            except Exception:
                pass

        # --- Duplicate near-identical columns (will be caught in cross-column) ---

    # ── Safe In-Place Fixes ───────────────────────────────────────────────────
    def _apply_safe_fixes(
        self, df: pd.DataFrame, col: str, dec: ColumnDecision
    ) -> None:
        """
        Apply only reversible, unambiguous fixes. Destructive decisions
        (drop, transform, impute) are recorded as recommendations and
        executed by the appropriate downstream stages.
        """
        # Currency — strip symbol characters so column becomes numeric
        if dec.semantic_type == SemanticType.CURRENCY and df[col].dtype == object:
            cleaned = (
                df[col].astype(str)
                .str.replace(r"[$£€¥,\s]", "", regex=True)
                .str.strip()
            )
            coerced = pd.to_numeric(cleaned, errors="coerce")
            # Only apply if we're not making things worse
            if coerced.notna().sum() >= df[col].notna().sum():
                df[col] = coerced
                dec.reasoning.append(
                    "Applied: stripped currency symbols ($£€) and converted to numeric."
                )

        # Numeric sentinels — replace known bad fill values
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = df[col].isin(self._NUM_SENTINELS)
            n = int(mask.sum())
            if n > 0:
                df.loc[mask, col] = np.nan
                dec.reasoning.append(
                    f"Applied: replaced {n} numeric sentinel values "
                    f"({sorted(self._NUM_SENTINELS)}) with NaN."
                )

        # Boolean normalisation
        if dec.semantic_type == SemanticType.BOOLEAN and df[col].dtype == object:
            mapping = {
                "yes": 1, "no": 0, "true": 1, "false": 0,
                "y": 1, "n": 0, "1": 1, "0": 0
            }
            df[col] = df[col].astype(str).str.lower().str.strip().map(mapping)
            dec.reasoning.append("Applied: normalised boolean strings to 0/1.")

    # ── Cross-Column Reasoning ────────────────────────────────────────────────
    def _cross_column_reasoning(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        report: BrainReport,
    ) -> None:
        numeric_df = df.select_dtypes(include="number")

        # Duplicate-value columns
        duplicated_cols: List[str] = []
        seen_hashes: Dict[str, str] = {}
        for col in df.columns:
            try:
                col_hash = hashlib.md5(
                    pd.util.hash_pandas_object(df[col], index=False).values.tobytes()
                ).hexdigest()
                if col_hash in seen_hashes:
                    duplicated_cols.append(col)
                    report.dataset_level_notes.append(
                        f"Column '{col}' is an exact duplicate of '{seen_hashes[col_hash]}'. "
                        "A data analyst would drop the redundant copy immediately."
                    )
                    if col in report.column_decisions:
                        report.column_decisions[col].should_drop = True
                        report.column_decisions[col].drop_reason = (
                            f"Exact duplicate of column '{seen_hashes[col_hash]}'."
                        )
                else:
                    seen_hashes[col_hash] = col
            except Exception:
                pass

        # Near-perfect correlation (redundant features)
        if len(numeric_df.columns) >= 2:
            try:
                corr = numeric_df.corr().abs()
                upper = corr.where(
                    np.triu(np.ones(corr.shape), k=1).astype(bool)
                )
                high_corr = [
                    (c1, c2, float(upper.loc[c1, c2]))
                    for c1 in upper.columns
                    for c2 in upper.index
                    if pd.notna(upper.loc[c1, c2]) and upper.loc[c1, c2] > 0.95
                    and c1 != target_col and c2 != target_col
                ]
                for c1, c2, corr_val in high_corr:
                    report.dataset_level_notes.append(
                        f"Features '{c1}' and '{c2}' are {corr_val:.2%} correlated — "
                        "nearly identical information. "
                        "Keeping one and flagging the other for removal (multicollinearity)."
                    )
                    if c2 in report.column_decisions:
                        report.column_decisions[c2].reasoning.append(
                            f"High correlation with '{c1}' ({corr_val:.2%}) — "
                            "redundant feature. One should be removed."
                        )
            except Exception:
                pass

        # Target leakage warning
        if target_col and target_col in df.columns:
            try:
                y = df[target_col]
                for col in numeric_df.columns:
                    if col == target_col:
                        continue
                    corr_with_target = abs(float(numeric_df[col].corr(y)))
                    if corr_with_target > 0.95:
                        report.dataset_level_notes.append(
                            f"⚠ DATA LEAKAGE RISK: '{col}' correlates {corr_with_target:.2%} "
                            f"with target '{target_col}'. This feature may be derived FROM "
                            "the target — using it would cause overfit."
                        )
                        if col in report.column_decisions:
                            report.column_decisions[col].violations.append(
                                f"Potential data leakage: {corr_with_target:.2%} correlation "
                                f"with target '{target_col}'."
                            )
            except Exception:
                pass

    # ── Target Suggestion ─────────────────────────────────────────────────────
    def _suggest_target(
        self, df: pd.DataFrame, report: BrainReport
    ) -> Optional[str]:
        """
        Suggest the most likely target column when none is provided.
        Priority: binary column > low-cardinality column matching known names.
        """
        target_hints = ["target", "label", "outcome", "churn", "fraud", "default",
                        "attrition", "survived", "class", "y", "response",
                        "converted", "approved", "rejected", "purchased"]
        # Check name-based hints first
        for col in df.columns:
            if col.lower() in target_hints:
                return col

        # Then find binary columns not already flagged for drop
        for col, dec in report.column_decisions.items():
            if dec.semantic_type == SemanticType.BINARY and not dec.should_drop:
                return col

        return None

    # ── Health Score ─────────────────────────────────────────────────────────
    def _score_health(self, df: pd.DataFrame, report: BrainReport) -> float:
        """
        0 = completely unusable, 100 = perfectly clean.
        Penalises: high nulls, many violations, many drops recommended,
                   high skew columns, constant columns.
        """
        score = 100.0
        decisions = list(report.column_decisions.values())
        if not decisions:
            return 0.0

        avg_null = float(df.isnull().mean().mean())
        score -= avg_null * 30                                         # nulls    -0 to -30

        drop_pct = sum(1 for d in decisions if d.should_drop) / len(decisions)
        score -= drop_pct * 20                                         # drops    -0 to -20

        violations = sum(len(d.violations) for d in decisions)
        score -= min(violations * 2, 20)                               # violations -0 to -20

        skewed = sum(1 for d in decisions if abs(d.skewness) > 2) / max(len(decisions), 1)
        score -= skewed * 10                                           # skew     -0 to -10

        return round(max(0.0, min(100.0, score)), 2)

    # ── Utility ───────────────────────────────────────────────────────────────
    @staticmethod
    def _looks_like_date(val: str) -> bool:
        DATE_RE = re.compile(
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|"       # 2024-01-15
            r"\d{1,2}[-/]\d{1,2}[-/]\d{4}|"           # 15/01/2024
            r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|"  # 15 Jan 2024
            r"jul|aug|sep|oct|nov|dec)\w*\s+\d{4})\b",
            re.IGNORECASE,
        )
        return bool(DATE_RE.search(val))

    @classmethod
    def from_config(cls, config: Dict) -> "AnalystBrain":
        return cls(config=config)
