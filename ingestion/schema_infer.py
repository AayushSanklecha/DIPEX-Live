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
    # Original 15
    "id", "age", "amount", "date", "category", "text",
    "phone", "email", "boolean", "zipcode", "percentage",
    "score", "count", "name", "unknown",
    # Extended 6 — common in web, IoT, logistics, finance data
    "url",           # http/https links, web endpoints
    "ip_address",    # IPv4 / IPv6 addresses
    "coordinates",   # latitude / longitude values (-90..90 / -180..180)
    "duration",      # time intervals (seconds, minutes, HH:MM:SS strings)
    "address",       # street/postal addresses (longer than text, structured)
    "currency_code", # ISO-4217 codes: USD, EUR, INR, GBP, JPY …
]

# ── NLP keyword mapping (column-name heuristics) ─────────────────────────────

_NLP_TAG_MAP: Dict[str, List[str]] = {
    # ── Identifiers ───────────────────────────────────────────────────────────
    "id": [
        r"\bid\b", r"_id$", r"^id_", r"uuid", r"guid", r"key",
        r"identifier", r"ref$", r"^ref_", r"serial", r"code$",
        r"accession", r"barcode", r"sku", r"asin", r"ean",
    ],
    # ── Temporal / Age ────────────────────────────────────────────────────────
    "age": [
        r"\bage\b", r"years?_?old", r"dob", r"birth", r"age_",
        r"_age$", r"yrs", r"tenure", r"seniority",
    ],
    # ── Financial Amounts ─────────────────────────────────────────────────────
    "amount": [
        r"amount", r"price", r"revenue", r"cost", r"salary",
        r"income", r"fee", r"balance", r"value", r"total",
        r"payment", r"spend", r"budget", r"expense", r"sales",
        r"profit", r"loss", r"margin", r"tax", r"wage",
        r"credit", r"debit", r"charge", r"invoice", r"fare",
        r"premium", r"subsidy", r"grant", r"bonus", r"dividend",
        r"voltage", r"current", r"power", r"energy",  # IoT / sensor
        r"temperature", r"pressure", r"flow", r"weight", r"mass",
        r"distance", r"length", r"height", r"width", r"depth",
        r"speed", r"velocity", r"concentration", r"dosage",
    ],
    # ── Dates / Timestamps ────────────────────────────────────────────────────
    "date": [
        r"date", r"time", r"_at$", r"_on$", r"timestamp",
        r"created", r"updated", r"modified", r"born", r"expires",
        r"start", r"end", r"period", r"year", r"month", r"day",
        r"datetime", r"recorded", r"reported", r"filed", r"posted",
    ],
    # ── Contact / PII ─────────────────────────────────────────────────────────
    "phone":   [r"phone", r"mobile", r"cell", r"tel", r"fax",
                r"contact_no", r"whatsapp"],
    "email":   [r"email", r"e_?mail", r"mail", r"inbox"],
    # ── People Names ──────────────────────────────────────────────────────────
    "name": [
        r"\bname\b", r"first_?name", r"last_?name", r"full_?name",
        r"username", r"user_?name", r"customer_name", r"author",
        r"owner", r"employee", r"vendor", r"supplier", r"contact",
        r"alias", r"display_name", r"label_name",
    ],
    # ── Geographies / Codes ───────────────────────────────────────────────────
    "zipcode":  [r"zip", r"postal", r"pincode", r"postcode", r"plz"],
    # ── Categories / Enumerations ─────────────────────────────────────────────
    "category": [
        r"category", r"type", r"class", r"group", r"segment",
        r"label", r"status", r"gender", r"region", r"country",
        r"city", r"state", r"department", r"division", r"sector",
        r"industry", r"channel", r"platform", r"tier", r"level",
        r"priority", r"mode", r"flag", r"tag", r"genre",
        r"brand", r"model", r"product", r"variant", r"colour",
        r"color", r"material", r"method", r"outcome", r"result",
        r"diagnosis", r"treatment", r"condition", r"fault",
    ],
    # ── Scores / Ratings ──────────────────────────────────────────────────────
    "score": [
        r"score", r"rating", r"rank", r"grade", r"gpa",
        r"fico", r"cibil", r"nps", r"satisfaction", r"quality",
        r"accuracy", r"precision", r"recall", r"f1",
        r"confidence", r"probability", r"weight", r"importance",
        r"sentiment", r"polarity",
    ],
    # ── Percentages / Rates ───────────────────────────────────────────────────
    "percentage": [
        r"pct", r"percent", r"rate", r"ratio", r"proportion",
        r"fraction", r"share", r"utilization", r"utilisation",
        r"efficiency", r"occupancy", r"coverage", r"completion",
        r"error_rate", r"default_rate", r"churn", r"conversion",
        r"growth", r"inflation", r"yield", r"apr", r"apy",
        r"humidity", r"moisture",  # sensor / environmental
    ],
    # ── Counts / Integers ─────────────────────────────────────────────────────
    "count": [
        r"count", r"num", r"number", r"qty", r"quantity",
        r"volume", r"n_", r"total_", r"_total$",
        r"clicks", r"views", r"visits", r"sessions", r"hits",
        r"orders", r"transactions", r"events", r"records",
        r"occurrences", r"frequency", r"attempts",
    ],
    # ── Boolean Flags ─────────────────────────────────────────────────────────
    "boolean": [
        r"^is_", r"^has_", r"^flag", r"active", r"enabled",
        r"verified", r"approved", r"valid", r"deleted", r"archived",
        r"exists", r"available", r"visible", r"public", r"primary",
        r"default", r"required", r"mandatory", r"allowed",
    ],
    # ── Free Text / Descriptions ──────────────────────────────────────────────
    "text": [
        r"description", r"notes?", r"comment", r"remarks", r"message",
        r"narrative", r"summary", r"body", r"content", r"text",
        r"review", r"feedback", r"reason", r"detail",
        r"bio",
    ],
    # ── Extended labels ───────────────────────────────────────────────────────
    "url": [
        r"url", r"link", r"href", r"endpoint", r"uri", r"website",
        r"domain", r"homepage", r"permalink", r"source_url", r"image_url",
        r"avatar", r"thumbnail",
    ],
    "ip_address": [
        r"ip", r"ip_addr", r"ipv4", r"ipv6", r"host", r"remote_addr",
        r"client_ip", r"server_ip", r"origin_ip",
    ],
    "coordinates": [
        r"lat", r"latitude", r"lng", r"lon", r"longitude",
        r"geo_x", r"geo_y", r"coord", r"x_coord", r"y_coord",
        r"northing", r"easting",
    ],
    "duration": [
        r"duration", r"elapsed", r"latency", r"response_time",
        r"uptime", r"downtime", r"ttl", r"timeout", r"interval",
        r"session_length", r"call_duration", r"watch_time",
    ],
    "address": [
        r"address", r"street", r"addr", r"location",
        r"building", r"house_no", r"flat", r"suite",
    ],
    "currency_code": [
        r"currency", r"ccy", r"iso_currency", r"currency_code",
        r"fx", r"forex", r"base_currency", r"quote_currency",
    ],
}

# Types where a column-name match is definitive (skip ML entirely)
_NLP_TIER_A = {
    "email", "phone", "date", "id", "boolean", "zipcode", "name",
    # Extended high-precision — regex patterns are unmistakable
    "url", "ip_address", "coordinates", "currency_code",
}

# Types where a column-name match is a strong hint (use when ML is uncertain)
_NLP_TIER_B = {
    "age", "amount", "score", "count", "percentage",
    "category", "text",
    # Extended medium-precision
    "duration", "address",
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

    # Extended pattern signals (for new labels)
    url_pattern = (
        float(str_vals.str.contains(r"https?://|www\.", na=False).mean())
        if is_str else 0.0
    )
    ip_pattern = (
        float(str_vals.str.match(
            r"^(\d{1,3}\.){3}\d{1,3}$|^([0-9a-fA-F]{1,4}:){2,7}", na=False
        ).mean())
        if is_str else 0.0
    )
    # Coordinates: numeric, bounded [-180, 180], high decimal precision
    coord_range = (
        float(((num_vals >= -180) & (num_vals <= 180)).all())
        if len(num_vals) > 0 else 0.0
    )
    coord_precision = (
        float((num_vals % 1 != 0).mean() > 0.8)   # most values are non-integer
        if len(num_vals) > 0 else 0.0
    )
    # Currency code: 3-char uppercase string
    currency_pattern = (
        float(str_vals.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7)
        if is_str else 0.0
    )

    return {
        "null_rate":         null_rate,
        "unique_rate":       unique_rate,
        "is_numeric":        is_numeric_f,
        "is_string":         is_string_f,
        "is_datetime":       is_datetime_f,
        "mean_val":          mean_val,
        "std_val":           std_val,
        "min_val":           min_val,
        "max_val":           max_val,
        "skew_val":          skew_val,
        "all_integer":       all_integer,
        "max_lt_200":        max_lt_200,
        "max_lt_1":          max_lt_1,
        "all_positive":      all_pos,
        "n_distinct":        n_distinct,
        "email_pattern":     email_pattern,
        "phone_pattern":     phone_pattern,
        "mean_str_len":      mean_str_len,
        "high_cardinality":  float(unique_rate > 0.9),
        "low_cardinality":   float(unique_rate < 0.05),
        # Extended features (used by heuristic; added to ML feature vector on retrain)
        "url_pattern":       url_pattern,
        "ip_pattern":        ip_pattern,
        "coord_range":       coord_range,
        "coord_precision":   coord_precision,
        "currency_pattern":  currency_pattern,
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
    """Rule-based semantic type inference (21-label). Used when ML is absent."""
    tags = _nlp_tags_from_name(col_name)
    if tags:
        return tags[0]

    f = _extract_column_features(series, col_name)

    # ── Pattern-first checks (highest signal strength) ────────────────────────
    if f["url_pattern"] > 0.7:
        return "url"
    if f["ip_pattern"] > 0.7:
        return "ip_address"
    if f["currency_pattern"] > 0.7:
        return "currency_code"
    if f["is_datetime"] == 1.0:
        return "date"
    if f["email_pattern"] > 0.7:
        return "email"
    if f["phone_pattern"] > 0.7:
        return "phone"

    # ── Coordinate detection ──────────────────────────────────────────────────
    # lat: -90..90, lon: -180..180, both high decimal precision
    if (
        f["is_numeric"] == 1.0
        and f["coord_range"] == 1.0
        and f["coord_precision"] == 1.0
        and f["min_val"] < 0   # coordinates span negative values
    ):
        return "coordinates"

    # ── Numeric rules ─────────────────────────────────────────────────────────
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

    # ── String rules ──────────────────────────────────────────────────────────
    if f["is_string"] == 1.0:
        if f["low_cardinality"]:
            return "category"
        # Duration strings (HH:MM:SS, 90s, "1h 30m")
        s_sample = series.dropna().astype(str)
        duration_pat = r"^\d+:\d{2}(:\d{2})?$|^\d+\s*(s|sec|min|h|hr|hours?)$"
        if len(s_sample) > 0 and s_sample.str.match(duration_pat, na=False).mean() > 0.5:
            return "duration"
        # Address strings tend to be long and contain digits mixed with words
        if f["mean_str_len"] > 20:
            addr_pat = r"\d+.*\b(st|street|rd|road|ave|avenue|blvd|ln|lane|dr|drive|nagar|colony)\b"
            if s_sample.str.contains(addr_pat, case=False, na=False, regex=True).mean() > 0.3:
                return "address"
        if f["mean_str_len"] > 60:
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
        Infer semantic type of a single column using a 3-layer cascade.

        Layer 1 — NLP column-name (ALL 15 labels, two tiers)
            Tier A — high-precision types (email/phone/date/id/boolean/
                      zipcode/name): return immediately, conf=0.95.
            Tier B — medium-precision types (age/amount/score/count/
                      percentage/category/text): record as nlp_hint.
                      Used in Layer 2 when ML is absent or uncertain.

        Layer 2 — ML model with confidence gate (0.45)
            RandomForestClassifier. Only accepted if conf >= 0.45 AND
            prediction is not 'unknown'. When ML is absent/uncertain and
            an NLP hint exists, the NLP hint is used (conf=0.75).

        Layer 3 — Heuristic rule engine
            Statistical inference. Last resort before 'unknown'.

        'unknown' is only returned when ALL three layers genuinely fail.
        """
        _ML_CONF_THRESHOLD = 0.45
        nlp_tags  = _nlp_tags_from_name(col_name)
        nlp_hint  = nlp_tags[0] if nlp_tags else None   # best NLP guess

        # ── Layer 1 Tier A: High-precision NLP — return immediately ───────────
        if nlp_hint and nlp_hint in _NLP_TIER_A:
            logger.debug(
                "SmartSchemaInferer [NLP-A] col='%s' -> '%s'",
                col_name, nlp_hint,
            )
            return {
                "semantic_type": nlp_hint,
                "confidence":    0.95,
                "method":        "nlp_name",
                "nlp_tags":      nlp_tags,
            }

        # ── Layer 2: ML model with confidence gate ────────────────────────────
        if self._model is not None:
            try:
                feats = _extract_column_features(series, col_name)
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

                if sem_type != "unknown" and conf >= _ML_CONF_THRESHOLD:
                    logger.debug(
                        "SmartSchemaInferer [ML] col='%s' -> '%s' (conf=%.2f)",
                        col_name, sem_type, conf,
                    )
                    return {
                        "semantic_type": sem_type,
                        "confidence":    round(conf, 4),
                        "method":        "ml",
                        "nlp_tags":      nlp_tags,
                    }

                # ML gave "unknown" or was below threshold.
                # If a Tier-B NLP hint exists, use it now instead of
                # going all the way to the heuristic engine.
                if nlp_hint and nlp_hint in _NLP_TIER_B:
                    logger.debug(
                        "SmartSchemaInferer [NLP-B] col='%s' ML uncertain "
                        "(gave '%s' conf=%.2f) -> NLP hint '%s'",
                        col_name, sem_type, conf, nlp_hint,
                    )
                    return {
                        "semantic_type": nlp_hint,
                        "confidence":    0.75,
                        "method":        "nlp_hint",
                        "nlp_tags":      nlp_tags,
                    }

                logger.debug(
                    "SmartSchemaInferer [ML] col='%s' gave '%s' conf=%.2f "
                    "— falling to heuristic.",
                    col_name, sem_type, conf,
                )

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SmartSchemaInferer ML inference failed for '%s': %s",
                    col_name, exc,
                )
                # ML crashed — if NLP Tier-B hint exists, use it
                if nlp_hint and nlp_hint in _NLP_TIER_B:
                    return {
                        "semantic_type": nlp_hint,
                        "confidence":    0.75,
                        "method":        "nlp_hint",
                        "nlp_tags":      nlp_tags,
                    }

        else:
            # No ML model loaded — if NLP Tier-B hint exists, use it
            # before running the heuristic engine.
            if nlp_hint and nlp_hint in _NLP_TIER_B:
                logger.debug(
                    "SmartSchemaInferer [NLP-B/no-model] col='%s' -> '%s'",
                    col_name, nlp_hint,
                )
                return {
                    "semantic_type": nlp_hint,
                    "confidence":    0.75,
                    "method":        "nlp_hint",
                    "nlp_tags":      nlp_tags,
                }

        # ── Layer 3: Statistical heuristic engine ─────────────────────────────
        heuristic_type = _heuristic_infer(series, col_name)

        # If heuristic still gives "unknown" and any NLP tag was found,
        # use the NLP tag as last resort before returning "unknown".
        if heuristic_type == "unknown" and nlp_hint:
            logger.debug(
                "SmartSchemaInferer [NLP-rescue] col='%s' -> '%s'",
                col_name, nlp_hint,
            )
            return {
                "semantic_type": nlp_hint,
                "confidence":    0.55,
                "method":        "nlp_rescue",
                "nlp_tags":      nlp_tags,
            }

        method = "heuristic" if self._model is None else "heuristic_fallback"
        return {
            "semantic_type": heuristic_type,
            "confidence":    0.70 if heuristic_type != "unknown" else 0.0,
            "method":        method,
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
