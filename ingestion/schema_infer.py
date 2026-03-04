"""
ingestion/schema_infer.py
--------------------------
Production-grade ML Semantic Type Classifier.

Architecture
------------
At runtime this module attempts to load a pre-trained RandomForestClassifier
exported from `colab/train_schema_classifier.ipynb`.

If the model artifact (`models/schema_classifier.pkl`) is absent the module
falls back to a deterministic heuristic engine — the pipeline never breaks.

Semantic Labels
---------------
  id, age, amount, date, category, text, phone, email, boolean, zipcode,
  percentage, score, count, name, unknown

Training
--------
See `colab/train_schema_classifier.ipynb`.  Exports:
    models/schema_classifier.pkl   — fitted RandomForestClassifier
    models/schema_label_encoder.pkl — fitted LabelEncoder

Usage
-----
    from ingestion.schema_infer import SmartSchemaInferer

    inferer = SmartSchemaInferer()
    result  = inferer.infer(df, column_name="age")
    # result = {"semantic_type": "age", "confidence": 0.92, "method": "ml"}

    enriched = inferer.enrich_schema(df, schema_dict)
    # enriched = {"age": {"dtype": "int64", "semantic_type": "age", "nlp_tags": [...], ...}}
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.ingestion.schema_infer")

# ── Constants ─────────────────────────────────────────────────────────────────

_MODEL_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")
_CLASSIFIER_PATH = os.path.join(_MODEL_DIR, "schema_classifier.pkl")
_ENCODER_PATH    = os.path.join(_MODEL_DIR, "schema_label_encoder.pkl")

SEMANTIC_LABELS = [
    "id", "age", "amount", "date", "category", "text",
    "phone", "email", "boolean", "zipcode", "percentage",
    "score", "count", "name", "unknown",
]

# ── NLP keyword mapping (column-name heuristics) ─────────────────────────────

_NLP_TAG_MAP: Dict[str, List[str]] = {
    "id":         [r"\bid\b", r"_id$", r"^id_", r"uuid", r"guid", r"key"],
    "age":        [r"\bage\b", r"years?_?old", r"dob"],
    "amount":     [r"amount", r"price", r"revenue", r"cost", r"salary",
                   r"income", r"fee", r"balance", r"value", r"total",
                   r"payment"],
    "date":       [r"date", r"time", r"_at$", r"_on$", r"timestamp",
                   r"created", r"updated", r"born"],
    "phone":      [r"phone", r"mobile", r"cell", r"tel"],
    "email":      [r"email", r"e_?mail", r"mail"],
    "name":       [r"\bname\b", r"first_?name", r"last_?name", r"full_?name",
                   r"username", r"user_?name"],
    "zipcode":    [r"zip", r"postal", r"pincode"],
    "category":   [r"category", r"type", r"class", r"group", r"segment",
                   r"label", r"status", r"gender", r"region", r"country",
                   r"city"],
    "score":      [r"score", r"rating", r"rank", r"grade", r"priority"],
    "percentage": [r"pct", r"percent", r"rate", r"ratio"],
    "count":      [r"count", r"num", r"number", r"qty", r"quantity",
                   r"volume", r"n_"],
    "boolean":    [r"^is_", r"^has_", r"^flag", r"active", r"enabled",
                   r"verified"],
}


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_column_features(series: pd.Series, col_name: str) -> Dict[str, float]:
    """
    Extract a numeric feature vector for a single column.
    These same 20 features must be used when training the Colab model.
    """
    s = series.dropna()
    n = max(len(s), 1)

    is_num   = pd.api.types.is_numeric_dtype(series)
    is_str   = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    is_dt    = pd.api.types.is_datetime64_any_dtype(series)

    num_vals = pd.to_numeric(s, errors="coerce").dropna() if not is_num else s
    str_vals = s.astype(str) if is_str else pd.Series([], dtype=str)

    null_rate      = series.isnull().mean()
    unique_rate    = series.nunique(dropna=True) / max(len(series), 1)
    is_numeric_f   = float(is_num)
    is_string_f    = float(is_str)
    is_datetime_f  = float(is_dt)

    # Numeric stats (safe defaults for non-numeric cols)
    mean_val  = float(num_vals.mean())  if len(num_vals) > 0 else 0.0
    std_val   = float(num_vals.std())   if len(num_vals) > 1 else 0.0
    min_val   = float(num_vals.min())   if len(num_vals) > 0 else 0.0
    max_val   = float(num_vals.max())   if len(num_vals) > 0 else 0.0
    skew_val  = float(num_vals.skew()) if len(num_vals) > 3 else 0.0

    # Binary checks
    all_integer   = float((num_vals == num_vals.astype(int)).all()) if len(num_vals) > 0 else 0.0
    max_lt_200    = float(max_val < 200) if len(num_vals) > 0 else 0.0
    max_lt_1      = float(max_val <= 1.0) if len(num_vals) > 0 else 0.0
    all_pos       = float((num_vals >= 0).all()) if len(num_vals) > 0 else 0.0
    n_distinct    = float(series.nunique(dropna=True))

    # String pattern signals
    email_pattern = float(str_vals.str.contains(r"@.*\.", na=False).mean()) if is_str else 0.0
    phone_pattern = float(str_vals.str.contains(r"^\+?\d[\d\s\-()]{7,}$",
                                                 na=False, regex=True).mean()) if is_str else 0.0
    mean_str_len  = float(str_vals.str.len().mean()) if is_str and len(str_vals) > 0 else 0.0

    return {
        "null_rate":     null_rate,
        "unique_rate":   unique_rate,
        "is_numeric":    is_numeric_f,
        "is_string":     is_string_f,
        "is_datetime":   is_datetime_f,
        "mean_val":      mean_val,
        "std_val":       std_val,
        "min_val":       min_val,
        "max_val":       max_val,
        "skew_val":      skew_val,
        "all_integer":   all_integer,
        "max_lt_200":    max_lt_200,
        "max_lt_1":      max_lt_1,
        "all_positive":  all_pos,
        "n_distinct":    n_distinct,
        "email_pattern": email_pattern,
        "phone_pattern": phone_pattern,
        "mean_str_len":  mean_str_len,
        "high_cardinality": float(unique_rate > 0.9),
        "low_cardinality":  float(unique_rate < 0.05),
    }


def _nlp_tags_from_name(col_name: str) -> List[str]:
    """Return matching semantic tag hints from column name keywords."""
    lower = col_name.lower().replace(" ", "_")
    tags: List[str] = []
    for tag, patterns in _NLP_TAG_MAP.items():
        for pat in patterns:
            if re.search(pat, lower):
                tags.append(tag)
                break
    return tags


# ── Heuristic fallback ─────────────────────────────────────────────────────────

def _heuristic_infer(series: pd.Series, col_name: str) -> str:
    """Rule-based semantic type inference. Used when ML model is absent."""
    tags = _nlp_tags_from_name(col_name)
    if tags:
        return tags[0]

    f = _extract_column_features(series, col_name)

    if f["is_datetime"] == 1.0:
        return "date"
    if f["email_pattern"] > 0.7:
        return "email"
    if f["phone_pattern"] > 0.7:
        return "phone"
    if f["is_numeric"] == 1.0:
        if f["max_lt_1"] and f["all_positive"]:
            return "percentage"
        if f["max_lt_200"] and f["all_integer"] and f["min_val"] >= 0:
            return "age"
        if f["high_cardinality"]:
            return "id"
        if f["all_integer"] and f["low_cardinality"]:
            return "count"
        return "amount"
    if f["is_string"] == 1.0:
        if f["low_cardinality"]:
            return "category"
        if f["mean_str_len"] > 40:
            return "text"
        return "name"
    return "unknown"


# ── Main class ─────────────────────────────────────────────────────────────────

class SmartSchemaInferer:
    """
    Production-grade semantic type classifier for DataFrame columns.

    Loads a pre-trained RandomForestClassifier if available; otherwise
    uses the deterministic heuristic engine as fallback.
    """

    def __init__(self) -> None:
        self._model: Any    = None
        self._encoder: Any  = None
        self._method: str   = "heuristic"
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load artifact from models/."""
        try:
            import joblib  # type: ignore
            if os.path.exists(_CLASSIFIER_PATH) and os.path.exists(_ENCODER_PATH):
                self._model   = joblib.load(_CLASSIFIER_PATH)
                self._encoder = joblib.load(_ENCODER_PATH)
                self._method  = "ml"
                logger.info("SmartSchemaInferer: ML model loaded from %s", _CLASSIFIER_PATH)
            else:
                logger.info(
                    "SmartSchemaInferer: model artifacts not found at %s — using heuristic fallback.",
                    _CLASSIFIER_PATH,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SmartSchemaInferer: model load failed (%s) — using heuristic.", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Infer semantic type of a single column.

        Returns
        -------
        {
          "semantic_type": str,
          "confidence":    float,    # 1.0 for heuristic, proba for ML
          "method":        str,      # "ml" | "heuristic"
          "nlp_tags":      List[str],
        }
        """
        nlp_tags = _nlp_tags_from_name(col_name)

        if self._model is not None:
            try:
                feats    = _extract_column_features(series, col_name)
                # Explicit key ordering — must match Colab training feature order exactly
                _FEAT_ORDER = [
                    "null_rate", "unique_rate", "is_numeric", "is_string",
                    "is_datetime", "mean_val", "std_val", "min_val", "max_val",
                    "skew_val", "all_integer", "max_lt_200", "max_lt_1",
                    "all_positive", "n_distinct", "email_pattern", "phone_pattern",
                    "mean_str_len", "high_cardinality", "low_cardinality",
                ]
                X        = np.array([feats[k] for k in _FEAT_ORDER]).reshape(1, -1)
                proba    = self._model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                sem_type = str(self._encoder.inverse_transform([pred_idx])[0])
                conf     = float(proba[pred_idx])
                return {
                    "semantic_type": sem_type,
                    "confidence":    round(conf, 4),
                    "method":        "ml",
                    "nlp_tags":      nlp_tags,
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("SmartSchemaInferer ML inference failed for '%s': %s", col_name, exc)

        # Fallback
        sem_type = _heuristic_infer(series, col_name)
        return {
            "semantic_type": sem_type,
            "confidence":    1.0,
            "method":        "heuristic",
            "nlp_tags":      nlp_tags,
        }

    def enrich_schema(
        self,
        df:          pd.DataFrame,
        schema_dict: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Enrich every column in schema_dict with semantic type, confidence, and NLP tags.

        Parameters
        ----------
        df          : DataFrame being processed
        schema_dict : {column_name: dtype_str}

        Returns
        -------
        {
          column_name: {
            "dtype":         str,
            "semantic_type": str,
            "confidence":    float,
            "method":        str,
            "nlp_tags":      List[str],
          }
        }
        """
        enriched: Dict[str, Dict[str, Any]] = {}
        for col, dtype in schema_dict.items():
            if col not in df.columns:
                enriched[col] = {
                    "dtype":         dtype,
                    "semantic_type": "unknown",
                    "confidence":    0.0,
                    "method":        "not_found",
                    "nlp_tags":      [],
                }
                continue
            result = self.infer(df[col], col)
            enriched[col] = {
                "dtype":         dtype,
                "semantic_type": result["semantic_type"],
                "confidence":    result["confidence"],
                "method":        result["method"],
                "nlp_tags":      result["nlp_tags"],
            }
        logger.info(
            "SmartSchemaInferer: enriched %d columns via %s.",
            len(enriched), self._method,
        )
        return enriched
