"""
validation/regulatory/finance_rules.py
----------------------------------------
Financial Reporting & Capital Markets Regulatory Rules for DIPEX.

Rules implemented (IFRS / SEC / MiFID II / Basel III):
  1. RevenueRecognitionRule     — Revenue must be non-negative; negative revenue
                                  requires an explicit credit-memo flag
  2. CapitalAdequacyRule        — Tier-1 capital ratio must meet Basel III minimum
  3. NetPositionLimitRule       — Net trading position must not breach position limits
  4. MarginCallThresholdRule    — Margin account must meet minimum maintenance margin
  5. SECFilingBoundsRule        — Financial statement figures must fall within
                                  SEC-defined materiality bounds
  6. DoubleEntryBalanceRule     — Debit and credit columns must net to zero
  7. FairValueHierarchyRule     — Level-3 fair value assets must not exceed IFRS 13 cap

All rules inherit BaseRegulatoryRule; violations carry human-readable remediations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_rule import BaseRegulatoryRule, RegulatoryViolation

logger = logging.getLogger(__name__)


class RevenueRecognitionRule(BaseRegulatoryRule):
    """
    IFRS 15: Revenue from Contracts with Customers.

    Revenue columns must be >= 0. Negative values are only allowable when a
    `credit_memo` flag column is present and set to True for that row.
    Regulatory basis: IFRS 15, ASC 606.
    """
    name = "revenue_recognition"
    domain = "finance"

    def __init__(
        self,
        revenue_columns: List[str],
        credit_memo_column: Optional[str] = None,
    ) -> None:
        self.revenue_columns = revenue_columns
        self.credit_memo_column = credit_memo_column

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        violations: List[RegulatoryViolation] = []

        for col in self.revenue_columns:
            if self._col_missing(col, df):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col]
            negative_mask = series < 0

            # Allow negatives if credit_memo flag is set
            if self.credit_memo_column and self.credit_memo_column in df.columns:
                negative_mask = negative_mask & ~df[self.credit_memo_column].astype(bool)

            bad = negative_mask.sum()
            if bad > 0:
                violations.append(RegulatoryViolation(
                    rule_name=self.name,
                    domain=self.domain,
                    severity="ERROR",
                    column=col,
                    offending_count=int(bad),
                    message=(
                        f"[IFRS 15] Column '{col}' has {bad} negative revenue "
                        f"value(s) without a corresponding credit memo indicator. "
                        f"Min value: {series.min():.4g}"
                    ),
                    remediation=(
                        f"Apply credit memo flag in '{self.credit_memo_column}' "
                        "for legitimate negative revenue, or correct data entry errors."
                    ),
                ))

        return violations


class CapitalAdequacyRule(BaseRegulatoryRule):
    """
    Basel III / RBI: Tier-1 Capital Adequacy Ratio (CAR) >= 8.0%.

    Computes Tier1_Capital / Risk_Weighted_Assets for each row and flags
    breaches. The default threshold aligns with BIS Basel III Pillar 1 minimum.
    """
    name = "capital_adequacy"
    domain = "finance"

    def __init__(
        self,
        tier1_col: str,
        rwa_col: str,
        min_car: float = 0.08,
    ) -> None:
        self.tier1_col = tier1_col
        self.rwa_col = rwa_col
        self.min_car = min_car

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.tier1_col, df) or self._col_missing(self.rwa_col, df):
            return []

        pairs = df[[self.tier1_col, self.rwa_col]].dropna()
        safe_rwa = pairs[self.rwa_col].replace(0, float("nan")).dropna()
        car = pairs.loc[safe_rwa.index, self.tier1_col] / safe_rwa
        breaches = (car < self.min_car).sum()

        if breaches == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="CRITICAL",
            column=f"{self.tier1_col}/{self.rwa_col}",
            offending_count=int(breaches),
            message=(
                f"[Basel III CAR] {breaches} entity(-ies) fall below the minimum "
                f"Tier-1 Capital Adequacy Ratio of {self.min_car:.1%}. "
                f"Minimum ratio observed: {float(car.min()):.2%}"
            ),
            remediation=(
                "Increase Tier-1 capital (retained earnings, equity issuance) "
                "or reduce risk-weighted assets to meet the Basel III floor."
            ),
        )]


class NetPositionLimitRule(BaseRegulatoryRule):
    """
    MiFID II / Dodd-Frank: Net trading positions must not exceed configured
    position limits. Applies to derivatives, commodity, and equity books.

    Regulatory basis: MiFID II Article 57, CFTC position limit rules.
    """
    name = "net_position_limit"
    domain = "finance"

    def __init__(
        self,
        position_column: str,
        max_long: float,
        max_short: float,
        instrument_column: Optional[str] = None,
    ) -> None:
        self.position_column = position_column
        self.max_long = max_long
        self.max_short = max_short
        self.instrument_column = instrument_column

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.position_column, df):
            return []

        series = df[self.position_column].dropna()
        violations: List[RegulatoryViolation] = []

        long_breaches  = (series > self.max_long).sum()
        short_breaches = (series < -self.max_short).sum()

        if long_breaches > 0:
            violations.append(RegulatoryViolation(
                rule_name=self.name,
                domain=self.domain,
                severity="ERROR",
                column=self.position_column,
                offending_count=int(long_breaches),
                message=(
                    f"[MiFID II] {long_breaches} position(s) exceed the maximum "
                    f"long limit of {self.max_long:,.0f}. "
                    f"Max observed: {float(series.max()):,.0f}"
                ),
                remediation=(
                    "Reduce long exposures to comply with regulatory position limits. "
                    "File a position limit exception report if required."
                ),
            ))

        if short_breaches > 0:
            violations.append(RegulatoryViolation(
                rule_name=self.name,
                domain=self.domain,
                severity="ERROR",
                column=self.position_column,
                offending_count=int(short_breaches),
                message=(
                    f"[MiFID II] {short_breaches} position(s) breach the maximum "
                    f"short limit of -{self.max_short:,.0f}. "
                    f"Min observed: {float(series.min()):,.0f}"
                ),
                remediation=(
                    "Cover short positions or apply for regulatory exemption."
                ),
            ))

        return violations


class MarginCallThresholdRule(BaseRegulatoryRule):
    """
    Exchange / broker margin rules: margin_balance must be >= maintenance_margin.

    When margin_balance < maintenance_margin, a margin call is triggered.
    FINRA Rule 4210; SEC Regulation T.
    """
    name = "margin_call_threshold"
    domain = "finance"

    def __init__(
        self,
        margin_balance_col: str,
        maintenance_margin_col: str,
    ) -> None:
        self.margin_balance_col = margin_balance_col
        self.maintenance_margin_col = maintenance_margin_col

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if (
            self._col_missing(self.margin_balance_col, df)
            or self._col_missing(self.maintenance_margin_col, df)
        ):
            return []

        pairs = df[[self.margin_balance_col, self.maintenance_margin_col]].dropna()
        deficient = (pairs[self.margin_balance_col] < pairs[self.maintenance_margin_col]).sum()

        if deficient == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="WARNING",
            column=f"{self.margin_balance_col}:{self.maintenance_margin_col}",
            offending_count=int(deficient),
            message=(
                f"[FINRA 4210] {deficient} account(s) below maintenance margin "
                "threshold — margin call required."
            ),
            remediation=(
                "Issue margin call notices within T+1. Accounts that do not meet "
                "the call within 5 business days must be liquidated per reg requirements."
            ),
        )]


class SECFilingBoundsRule(BaseRegulatoryRule):
    """
    SEC Regulation S-X: Financial statement line items must fall within
    plausible materiality bounds to flag potential misstatements.

    Checks both lower (negative assets = suspicious) and upper bounds.
    SEC materiality threshold: typically 5% of total revenue/assets.
    """
    name = "sec_filing_bounds"
    domain = "finance"

    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        severity: str = "WARNING",
    ) -> None:
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.severity = severity

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.column, df):
            return []

        series = df[self.column].dropna()
        violations: List[RegulatoryViolation] = []

        if self.min_value is not None:
            below = (series < self.min_value).sum()
            if below > 0:
                violations.append(RegulatoryViolation(
                    rule_name=f"{self.name}_lower",
                    domain=self.domain,
                    severity=self.severity,
                    column=self.column,
                    offending_count=int(below),
                    message=(
                        f"[SEC Reg S-X] {below} value(s) in '{self.column}' "
                        f"below materiality floor of {self.min_value:,.2f}."
                    ),
                    remediation="Review for sign errors or unreported adjustments.",
                ))

        if self.max_value is not None:
            above = (series > self.max_value).sum()
            if above > 0:
                violations.append(RegulatoryViolation(
                    rule_name=f"{self.name}_upper",
                    domain=self.domain,
                    severity=self.severity,
                    column=self.column,
                    offending_count=int(above),
                    message=(
                        f"[SEC Reg S-X] {above} value(s) in '{self.column}' "
                        f"exceed materiality ceiling of {self.max_value:,.2f}."
                    ),
                    remediation=(
                        "Verify against source financial statements. "
                        "May indicate a scale error (thousands vs millions)."
                    ),
                ))

        return violations


class DoubleEntryBalanceRule(BaseRegulatoryRule):
    """
    Double-entry accounting: sum(debits) must equal sum(credits) per transaction.

    Groups by `transaction_id_col` and checks the net sum of `amount_col`
    (signed: debits positive, credits negative) is zero (within tolerance).
    Regulatory basis: GAAP / IFRS accounting standards.
    """
    name = "double_entry_balance"
    domain = "finance"

    def __init__(
        self,
        amount_col: str,
        transaction_id_col: str,
        tolerance: float = 0.01,
    ) -> None:
        self.amount_col = amount_col
        self.transaction_id_col = transaction_id_col
        self.tolerance = tolerance

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if (
            self._col_missing(self.amount_col, df)
            or self._col_missing(self.transaction_id_col, df)
        ):
            return []

        try:
            net = df.groupby(self.transaction_id_col)[self.amount_col].sum()
            imbalanced = (net.abs() > self.tolerance).sum()
        except Exception:
            return []

        if imbalanced == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="ERROR",
            column=self.amount_col,
            offending_count=int(imbalanced),
            message=(
                f"[IFRS/GAAP] {imbalanced} transaction(s) fail the double-entry "
                f"balance check (tolerance ±{self.tolerance}). "
                "Net position is non-zero — ledger is out of balance."
            ),
            remediation=(
                "Reconcile entries against source journal. Verify all debit/credit "
                "pairs are captured. Check for missing contra entries."
            ),
        )]


class FairValueHierarchyRule(BaseRegulatoryRule):
    """
    IFRS 13 / ASC 820: Level-3 fair value assets (unobservable inputs) must
    not exceed a configurable threshold as a percentage of total fair value.

    Regulatory basis: IFRS 13, ASC 820 (Fair Value Measurement).
    """
    name = "fair_value_hierarchy"
    domain = "finance"

    def __init__(
        self,
        level3_col: str,
        total_fair_value_col: str,
        max_level3_ratio: float = 0.20,
    ) -> None:
        self.level3_col = level3_col
        self.total_fair_value_col = total_fair_value_col
        self.max_level3_ratio = max_level3_ratio

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if (
            self._col_missing(self.level3_col, df)
            or self._col_missing(self.total_fair_value_col, df)
        ):
            return []

        pairs = df[[self.level3_col, self.total_fair_value_col]].dropna()
        safe_total = pairs[self.total_fair_value_col].replace(0, float("nan")).dropna()
        ratio = pairs.loc[safe_total.index, self.level3_col] / safe_total
        breaches = (ratio > self.max_level3_ratio).sum()

        if breaches == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="WARNING",
            column=f"{self.level3_col}/{self.total_fair_value_col}",
            offending_count=int(breaches),
            message=(
                f"[IFRS 13] {breaches} entity(-ies) have Level-3 fair value assets "
                f"exceeding {self.max_level3_ratio:.0%} of total fair value. "
                f"Max ratio observed: {float(ratio.max()):.1%}"
            ),
            remediation=(
                "Increase observable market inputs to reclassify Level-3 assets "
                "to Level-2 where possible. Enhance disclosures on unobservable inputs."
            ),
        )]
