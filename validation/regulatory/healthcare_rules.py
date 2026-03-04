"""
validation/regulatory/healthcare_rules.py
-------------------------------------------
Healthcare domain regulatory rules.

Rules implemented:
  1. AgeRangeRule           — patient age within physiological bounds [0, 130]
  2. VitalSignsRule         — HR, SBP, DBP, SpO2, temperature within clinical ranges
  3. DiagnosisCodeFormatRule — ICD-10-CM code format validation
  4. PHIPresenceRule        — detects raw PII patterns (SSN, DOB) in unexpected columns
"""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_rule import BaseRegulatoryRule, RegulatoryViolation

logger = logging.getLogger(__name__)

# ICD-10-CM pattern: Letter + 2 digits, optionally followed by 1–4 alphanumeric chars
_ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$", re.IGNORECASE)

# Common PHI patterns for detection
_SSN_PATTERN  = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DOB_PATTERN  = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


class AgeRangeRule(BaseRegulatoryRule):
    """
    Patient age must be within physiologically plausible bounds.
    Regulatory basis: HL7 FHIR R4, HIPAA data quality standards.
    """
    name = "age_range"
    domain = "healthcare"

    def __init__(
        self,
        age_column: str = "age",
        min_age: float = 0.0,
        max_age: float = 130.0,
    ) -> None:
        self.age_column = age_column
        self.min_age = min_age
        self.max_age = max_age

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.age_column, df):
            return []

        series = df[self.age_column].dropna()
        below = (series < self.min_age).sum()
        above = (series > self.max_age).sum()
        bad = below + above

        if bad == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="ERROR",
            column=self.age_column,
            offending_count=int(bad),
            message=(
                f"[Healthcare] Column '{self.age_column}': {bad} record(s) outside "
                f"physiological age bounds [{self.min_age}, {self.max_age}]. "
                f"Below min: {below}, above max: {above}. "
                f"Range observed: [{series.min():.1f}, {series.max():.1f}]."
            ),
            remediation=(
                "Review patient registration data for data-entry errors. "
                "Check for age calculated in different units (months vs years)."
            ),
        )]


class VitalSignsRule(BaseRegulatoryRule):
    """
    Clinical vital signs must remain within established physiological bounds.

    Default bounds (configurable per instance):
      Heart Rate (HR):        [20, 300] bpm
      Systolic BP (SBP):      [50, 300] mmHg
      Diastolic BP (DBP):     [20, 200] mmHg
      SpO2 (oxygen sat.):     [50, 100] %
      Temperature:            [25.0, 45.0] °C

    Regulatory basis: SNOMED CT, HL7 FHIR Observation resource constraints.
    """
    name = "vital_signs"
    domain = "healthcare"

    _DEFAULT_BOUNDS: Dict[str, Dict[str, float]] = {
        "heart_rate":    {"min": 20,   "max": 300},
        "systolic_bp":   {"min": 50,   "max": 300},
        "diastolic_bp":  {"min": 20,   "max": 200},
        "spo2":          {"min": 50,   "max": 100},
        "temperature":   {"min": 25.0, "max": 45.0},
    }

    def __init__(self, column_bounds: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        # Caller can override specific columns; defaults fill the rest
        self.column_bounds: Dict[str, Dict[str, float]] = {
            **self._DEFAULT_BOUNDS,
            **(column_bounds or {}),
        }

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        violations: List[RegulatoryViolation] = []

        for col, bounds in self.column_bounds.items():
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            bad_low  = (series < bounds["min"]).sum()
            bad_high = (series > bounds["max"]).sum()
            bad = bad_low + bad_high

            if bad > 0:
                violations.append(RegulatoryViolation(
                    rule_name=self.name,
                    domain=self.domain,
                    severity="ERROR",
                    column=col,
                    offending_count=int(bad),
                    message=(
                        f"[Vitals] '{col}': {bad} value(s) outside clinical bounds "
                        f"[{bounds['min']}, {bounds['max']}]. "
                        f"Below min: {bad_low}, above max: {bad_high}."
                    ),
                    remediation=(
                        f"Validate '{col}' measurement units and sensor calibration. "
                        "Cross-check against source EHR records."
                    ),
                ))

        return violations


class DiagnosisCodeFormatRule(BaseRegulatoryRule):
    """
    ICD-10-CM diagnosis codes must conform to the standard format:
    one letter + two digits + optional decimal + up to 4 alphanumeric chars.
    Example valid: "E11.9", "Z00", "M79.3"

    Regulatory basis: CMS ICD-10-CM Official Guidelines, WHO ICD-10 standard.
    """
    name = "diagnosis_code_format"
    domain = "healthcare"

    def __init__(self, diagnosis_columns: Optional[List[str]] = None) -> None:
        self.diagnosis_columns: List[str] = diagnosis_columns or ["diagnosis_code", "icd_code"]

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        violations: List[RegulatoryViolation] = []

        for col in self.diagnosis_columns:
            if self._col_missing(col, df):
                continue

            series = df[col].dropna().astype(str)
            invalid = series[~series.str.match(_ICD10_PATTERN)]

            if not invalid.empty:
                sample = invalid.head(5).tolist()
                violations.append(RegulatoryViolation(
                    rule_name=self.name,
                    domain=self.domain,
                    severity="ERROR",
                    column=col,
                    offending_count=len(invalid),
                    message=(
                        f"[ICD-10] Column '{col}': {len(invalid)} code(s) do not match "
                        f"the ICD-10-CM format. Sample invalid: {sample}"
                    ),
                    remediation=(
                        "Map codes against the current CMS ICD-10-CM code set. "
                        "Reject free-text diagnosis entries at point of entry."
                    ),
                ))

        return violations


class PHIPresenceRule(BaseRegulatoryRule):
    """
    Detects raw Protected Health Information (PHI) patterns in unexpected
    free-text columns (SSN, date-of-birth, email addresses).

    This is a CRITICAL severity rule — raw PHI in ML datasets is a HIPAA
    violation and must halt the pipeline immediately.

    Regulatory basis: HIPAA Safe Harbor (45 CFR §164.514(b)).
    """
    name = "phi_presence"
    domain = "healthcare"

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        allowed_phi_columns: Optional[List[str]] = None,
    ) -> None:
        self.text_columns: Optional[List[str]] = text_columns
        self.allowed_phi_columns: List[str] = allowed_phi_columns or []

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        violations: List[RegulatoryViolation] = []

        # Default to all object-type columns if none specified
        check_cols = self.text_columns or list(
            df.select_dtypes(include=["object", "string"]).columns
        )

        phi_patterns = {
            "SSN":   _SSN_PATTERN,
            "DOB":   _DOB_PATTERN,
            "Email": _EMAIL_PATTERN,
        }

        for col in check_cols:
            if col in self.allowed_phi_columns or col not in df.columns:
                continue

            series = df[col].dropna().astype(str)

            for phi_type, pattern in phi_patterns.items():
                detected = series[series.str.contains(pattern, regex=True)]
                if not detected.empty:
                    violations.append(RegulatoryViolation(
                        rule_name=self.name,
                        domain=self.domain,
                        severity="CRITICAL",
                        column=col,
                        offending_count=len(detected),
                        message=(
                            f"[HIPAA § 164.514(b)] Column '{col}' contains "
                            f"{len(detected)} raw {phi_type} pattern(s). "
                            "PHI must be de-identified before ML processing."
                        ),
                        remediation=(
                            f"Apply HIPAA Safe Harbor de-identification to '{col}'. "
                            f"Remove or hash all {phi_type} patterns. "
                            "Do not ingest identifiable patient data into the ML pipeline."
                        ),
                    ))

        return violations
