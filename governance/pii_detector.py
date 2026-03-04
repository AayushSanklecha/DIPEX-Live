"""
governance/pii_detector.py
---------------------------
Production-grade PII (Personally Identifiable Information) Detector.

Architecture
------------
Two-tier detection, always with graceful fallback:

  Tier 1 — ML NER:
    Loads a spaCy NER model (trained in Colab on PII labelled data or uses
    a downloaded spaCy model for named-entity recognition).
    Detects: PERSON, EMAIL, PHONE, ORG, GPE, DATE, MONEY

  Tier 2 — Regex Patterns:
    Always runs in parallel as a cross-check.
    Covers: email, phone (E.164 + generic), SSN, credit card,
            IP address, US ZIP code, date-of-birth patterns.

Column-level decision: a column is flagged as PII if either tier detects
a PII signal in > 5 % of sampled values.

Colab Training
--------------
See colab/train_pii_ner.ipynb
Exports: models/pii_ner_model/  (spaCy serialised model directory)

Usage
-----
    from governance.pii_detector import PIIDetector

    detector = PIIDetector()
    report = detector.scan(df, sample_n=200)
    # report = {
    #   "pii_columns": {"email_col": ["email"], "name_col": ["PERSON"]},
    #   "safe_columns": ["amount", "date"],
    #   "method": "regex+spacy",
    # }
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("dipex.governance.pii_detector")

# ── Regex patterns ────────────────────────────────────────────────────────────

_PATTERNS: Dict[str, re.Pattern] = {
    "email":       re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    "phone":       re.compile(
        r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b"
    ),
    "ssn":         re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "us_zip":      re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "dob":         re.compile(
        r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b"
    ),
    "passport":    re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

_NER_PII_LABELS = {"PERSON", "GPE", "ORG", "DATE", "MONEY", "EMAIL"}
_PII_FLAG_THRESHOLD = 0.05   # flag column if > 5 % of sampled values contain PII


class PIIDetector:
    """
    Column-level PII scanner combining ML NER and regex patterns.
    """

    def __init__(self) -> None:
        self._nlp: Any   = None
        self._method: str = "regex_only"
        self._load_ner()

    def _load_ner(self) -> None:
        """Attempt to load spaCy NER model (Colab-trained or pre-trained)."""
        import os
        _CUSTOM_MODEL_DIR = os.path.join(
            os.path.dirname(__file__), "..", "models", "pii_ner_model"
        )
        try:
            import spacy  # type: ignore
            if os.path.exists(_CUSTOM_MODEL_DIR):
                self._nlp    = spacy.load(_CUSTOM_MODEL_DIR)
                self._method = "regex+custom_ner"
                logger.info("PIIDetector: loaded custom spaCy NER from %s", _CUSTOM_MODEL_DIR)
            else:
                # Try the standard English model
                self._nlp    = spacy.load("en_core_web_sm")
                self._method = "regex+spacy_sm"
                logger.info("PIIDetector: using en_core_web_sm for NER.")
        except (ImportError, OSError):
            logger.info("PIIDetector: spaCy not available — regex-only mode.")

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        df:       pd.DataFrame,
        sample_n: int = 200,
    ) -> Dict[str, Any]:
        """
        Scan all string columns in df for PII.

        Parameters
        ----------
        df       : DataFrame to scan
        sample_n : Max rows to sample per column (for performance)

        Returns
        -------
        {
          "pii_columns":  {col: [pii_types]},   # columns with detected PII
          "safe_columns": [col],
          "method":       str,
          "details":      {col: {pii_type: count}},
        }
        """
        str_cols     = df.select_dtypes(include=["object", "string"]).columns.tolist()
        pii_columns: Dict[str, List[str]] = {}
        safe_columns: List[str] = []
        details:      Dict[str, Dict[str, int]] = {}

        for col in str_cols:
            sample   = df[col].dropna().astype(str)
            if len(sample) > sample_n:
                sample = sample.sample(sample_n, random_state=42)

            pii_hits: Dict[str, int] = {}

            # Detect if the column is purely numeric strings
            # (i.e., all values match ^\d+(\.\d+)?$) — these are financial amounts,
            # not ZIP codes or SSNs, so skip those numeric-only PII patterns.
            _all_numeric = sample.str.match(r"^\d+(\.\d+)?$").all()
            _NUMERIC_ONLY_PATTERNS = {"us_zip", "phone"}

            # ── Regex scan ────────────────────────────────────────────────
            for pii_type, pat in _PATTERNS.items():
                # Skip zip/phone patterns on all-numeric columns (financial amounts)
                if _all_numeric and pii_type in _NUMERIC_ONLY_PATTERNS:
                    continue
                hits = int(sample.str.contains(pat, na=False).sum())
                if hits > 0:
                    pii_hits[pii_type] = hits

            # ── NER scan (if spaCy available) ────────────────────────────
            if self._nlp is not None:
                try:
                    text     = " | ".join(sample.values[:50])  # limit tokens
                    doc      = self._nlp(text)
                    for ent in doc.ents:
                        if ent.label_ in _NER_PII_LABELS:
                            pii_hits[f"NER:{ent.label_}"] = pii_hits.get(f"NER:{ent.label_}", 0) + 1
                except Exception:  # noqa: BLE001
                    pass

            # ── Decision: flag if any type exceeds threshold ───────────────
            total = max(len(sample), 1)
            flagged_types = [
                t for t, cnt in pii_hits.items()
                if cnt / total >= _PII_FLAG_THRESHOLD
            ]

            if flagged_types:
                pii_columns[col] = flagged_types
                details[col]     = pii_hits
            else:
                safe_columns.append(col)

        logger.info(
            "PIIDetector [%s]: %d PII columns, %d safe, across %d string columns.",
            self._method, len(pii_columns), len(safe_columns), len(str_cols),
        )
        return {
            "pii_columns":  pii_columns,
            "safe_columns": safe_columns,
            "method":       self._method,
            "details":      details,
        }

    def mask(
        self,
        df:     pd.DataFrame,
        report: Dict[str, Any],
        mask_char: str = "***",
    ) -> pd.DataFrame:
        """
        Mask all detected PII columns with `mask_char`.
        Returns a copy of df with PII columns replaced.
        """
        df = df.copy()
        for col in report.get("pii_columns", {}):
            if col in df.columns:
                df[col] = mask_char
                logger.info("PIIDetector: masked column '%s'.", col)
        return df
