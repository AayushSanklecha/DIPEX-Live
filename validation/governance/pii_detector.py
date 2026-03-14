"""
validation/governance/pii_detector.py
---------------------------------------
Detects Personally Identifiable Information (PII) within a DataFrame's text columns.

Uses fast regex patterns to detect:
    - Email Addresses
    - SSNs (US Social Security Numbers)
    - Credit Card Numbers (Luhn check not strictly enforced for speed, identifies 13-19 digits)
    - Phone Numbers (US/International standard formats)

Returns a DataFrame mask of detections.
"""

import logging
import re
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger("dipex.validation.governance.pii_detector")

# Comprehensive PII Regex Patterns
PII_PATTERNS = {
    "Email": re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
    "SSN": re.compile(
        r"\b(?!000)(?!666)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    ),
    "CreditCard": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b"
    ),
    "Phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    ),
    "IPAddress": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    "ICD10": re.compile(
        r"\b[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b"
    ),
    "IBAN": re.compile(
        r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}(?:[A-Z0-9]?){0,16}\b"
    ),
    "Swift": re.compile(
        r"\b[A-Z]{6}[A-Z2-9][A-NP-Z0-9](?:[A-Z0-9]{3})?\b"
    ),
}

# Static Redact Values
REDACTION_MASKS = {
    "Email": "[REDACTED_EMAIL]",
    "SSN": "[REDACTED_SSN]",
    "CreditCard": "[REDACTED_CC]",
    "Phone": "[REDACTED_PHONE]",
    "IPAddress": "[REDACTED_IP]",
    "ICD10": "[REDACTED_ICD10]",
    "IBAN": "[REDACTED_IBAN]",
    "Swift": "[REDACTED_SWIFT]",
}

class PIIDetector:
    """
    Scans a DataFrame's text columns for supported PII entities.
    """

    def __init__(self, targets: List[str] = None):
        """
        :param targets: List of PII types to detect. Defaults to all in PII_PATTERNS.
        """
        self.targets = targets or list(PII_PATTERNS.keys())
        self.patterns = {k: v for k, v in PII_PATTERNS.items() if k in self.targets}
        self.masks = {k: v for k, v in REDACTION_MASKS.items() if k in self.targets}

    def detect(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Scans all string/object columns for PII.
        Returns a dictionary detailing the number of hits per column per PII type.
        
        Example Return:
        {
            "user_email": {"Email": 150},
            "notes": {"Phone": 12, "SSN": 1}
        }
        """
        report = {}
        str_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in str_cols:
            col_series = df[col].dropna().astype(str)
            if col_series.empty:
                continue
                
            col_hits = {}
            for pii_type, pattern in self.patterns.items():
                # Count rows that contain at least one match
                hits = col_series.str.contains(pattern, regex=True).sum()
                if hits > 0:
                    col_hits[pii_type] = int(hits)
            
            if col_hits:
                report[col] = col_hits

        return report

    def redact(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        Clones the DataFrame, runs the regex patterns on text columns, and inline REDACTS
        any matching substrings with the corresponding [REDACTED_TYPE] mask.

        Returns:
            - The cleansed DataFrame
            - The detection report (same as .detect())
        """
        cleansed_df = df.copy()
        report = {}
        str_cols = cleansed_df.select_dtypes(include=["object", "string"]).columns

        for col in str_cols:
            col_series = cleansed_df[col].dropna().astype(str)
            if col_series.empty:
                continue

            col_hits = {}
            for pii_type, pattern in self.patterns.items():
                
                # Count matches for the report
                hits = col_series.str.contains(pattern, regex=True).sum()
                if hits > 0:
                    col_hits[pii_type] = int(hits)
                    # Replace the matches in the cleansed DataFrame
                    mask = self.masks[pii_type]
                    # We only update the rows that are not null
                    cleansed_df.loc[col_series.index, col] = col_series.str.replace(pattern, mask, regex=True)
                    # Update col_series so subsequent passes (e.g., Phone then Email) work correctly
                    col_series = cleansed_df.loc[col_series.index, col].astype(str)

            if col_hits:
                report[col] = col_hits

        return cleansed_df, report
