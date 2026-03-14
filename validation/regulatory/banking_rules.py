"""
validation/regulatory/banking_rules.py
----------------------------------------
Banking and financial domain regulatory rules.

Rules implemented:
  1. PositiveAmountRule              — monetary amount columns must be > 0
  2. AMLThresholdRule                — flags transactions at/above the AML reporting limit
  3. LoanRatioRule                   — loan-to-value ratio must remain within safe bounds
  4. RepaymentConsistencyRule        — repayment amount must not exceed outstanding balance
  5. SuspiciousTransactionPatternRule — velocity spike detection (FATF Rec. 20)
  6. CurrencyConcentrationRule       — BCBS239 concentration risk by currency
"""

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_rule import BaseRegulatoryRule, RegulatoryViolation

logger = logging.getLogger(__name__)


class PositiveAmountRule(BaseRegulatoryRule):
    """
    All monetary amount columns (e.g. transaction_amount, loan_amount, fee)
    must contain strictly positive values.

    Regulatory basis: ISO 20022, IFRS 9 reporting standards.
    """
    name = "positive_amount"
    domain = "banking"

    def __init__(self, amount_columns: List[str], allow_zero: bool = False) -> None:
        self.amount_columns = amount_columns
        self.allow_zero = allow_zero

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        violations: List[RegulatoryViolation] = []
        operator_desc = ">= 0" if self.allow_zero else "> 0"

        for col in self.amount_columns:
            if self._col_missing(col, df):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            bad = (series < 0).sum() if self.allow_zero else (series <= 0).sum()

            if bad > 0:
                violations.append(RegulatoryViolation(
                    rule_name=self.name,
                    domain=self.domain,
                    severity="ERROR",
                    column=col,
                    offending_count=int(bad),
                    message=(
                        f"[Banking] Column '{col}' must be {operator_desc}, "
                        f"but {bad} record(s) violate this constraint. "
                        f"Min observed: {series.min():.4g}"
                    ),
                    remediation=(
                        f"Review '{col}' for data entry errors, reversals, "
                        "or missing credit/debit sign conventions."
                    ),
                ))

        return violations


class AMLThresholdRule(BaseRegulatoryRule):
    """
    Anti-Money Laundering (AML): flags transactions at or above the
    regulatory reporting threshold (default: 10,000 in local currency units).

    This is a WARNING — flagged records require human review, not auto-reject.
    Regulatory basis: FATF Recommendation 10, PMLA.
    """
    name = "aml_threshold"
    domain = "banking"

    def __init__(
        self,
        amount_column: str,
        threshold: float = 10_000.0,
        currency_column: Optional[str] = None,
    ) -> None:
        self.amount_column = amount_column
        self.threshold = threshold
        self.currency_column = currency_column

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.amount_column, df):
            return []

        series = df[self.amount_column].dropna()
        flagged = (series >= self.threshold).sum()

        if flagged == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="WARNING",
            column=self.amount_column,
            offending_count=int(flagged),
            message=(
                f"[AML] {flagged} transaction(s) at or above the AML "
                f"reporting threshold of {self.threshold:,.2f}. "
                "These require manual SAR review."
            ),
            remediation=(
                "Flag these transactions for Suspicious Activity Report (SAR) "
                "submission within the regulatory window (typically 30 days)."
            ),
        )]


class LoanRatioRule(BaseRegulatoryRule):
    """
    Loan-to-Value (LTV) ratio must be within configurable bounds.
    Regulatory basis: Basel III capital adequacy, RBI prudential norms.
    """
    name = "loan_ratio"
    domain = "banking"

    def __init__(
        self,
        loan_col: str,
        value_col: str,
        max_ltv: float = 0.90,
        severity: str = "ERROR",
    ) -> None:
        self.loan_col = loan_col
        self.value_col = value_col
        self.max_ltv = max_ltv
        self.severity = severity

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.loan_col, df) or self._col_missing(self.value_col, df):
            return []

        pair = df[[self.loan_col, self.value_col]].dropna()
        safe_value = pair[self.value_col].replace(0, float("nan")).dropna()
        ltv = pair.loc[safe_value.index, self.loan_col] / safe_value
        breaches = (ltv > self.max_ltv).sum()

        if breaches == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity=self.severity,
            column=f"{self.loan_col}/{self.value_col}",
            offending_count=int(breaches),
            message=(
                f"[Basel III/LTV] {breaches} loan(s) exceed the maximum LTV "
                f"ratio of {self.max_ltv:.0%}. "
                f"Max observed ratio: {float(ltv.max()):.2%}"
            ),
            remediation=(
                "Obtain additional collateral or reduce loan exposure "
                "to bring LTV within regulatory limits."
            ),
        )]


class RepaymentConsistencyRule(BaseRegulatoryRule):
    """
    Repayment amount must not exceed the outstanding loan balance.
    Overpayment indicates data inconsistency or processing errors.
    """
    name = "repayment_consistency"
    domain = "banking"

    def __init__(self, repayment_col: str, balance_col: str) -> None:
        self.repayment_col = repayment_col
        self.balance_col = balance_col

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.repayment_col, df) or self._col_missing(self.balance_col, df):
            return []

        pair = df[[self.repayment_col, self.balance_col]].dropna()
        bad = (pair[self.repayment_col] > pair[self.balance_col]).sum()

        if bad == 0:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="ERROR",
            column=f"{self.repayment_col}:{self.balance_col}",
            offending_count=int(bad),
            message=(
                f"[Data Integrity] {bad} record(s) have repayment amount > "
                f"outstanding balance — indicating likely data corruption."
            ),
            remediation=(
                "Reconcile repayment records against the core banking "
                "system. Check for sign-convention errors or double-posting."
            ),
        )]


class SuspiciousTransactionPatternRule(BaseRegulatoryRule):
    """
    FATF Recommendation 20 — Reporting of suspicious transactions.

    Detects velocity spikes: accounts/entities with an unusually high number
    of transactions within a single day (or time period), which may indicate
    structuring (smurfing), layering, or automated fraud patterns.

    Regulatory basis: FATF Recommendation 20, USA PATRIOT Act §326, BSA.
    """
    name = "suspicious_transaction_pattern"
    domain = "banking"

    def __init__(
        self,
        transaction_id_column: str = "account_id",
        timestamp_column: Optional[str] = "transaction_date",
        max_transactions_per_day: int = 50,
    ) -> None:
        self.transaction_id_column = transaction_id_column
        self.timestamp_column = timestamp_column
        self.max_transactions_per_day = max_transactions_per_day

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.transaction_id_column, df):
            return []

        try:
            if self.timestamp_column and self.timestamp_column in df.columns:
                df_copy = df[[self.transaction_id_column, self.timestamp_column]].copy()
                df_copy["_date"] = pd.to_datetime(
                    df_copy[self.timestamp_column], errors="coerce"
                ).dt.date
                counts = (
                    df_copy.groupby([self.transaction_id_column, "_date"])
                    .size()
                    .reset_index(name="_count")
                )
                flagged_groups = counts[counts["_count"] > self.max_transactions_per_day]
                flagged_count = int(flagged_groups[self.transaction_id_column].nunique())
            else:
                counts = df[self.transaction_id_column].value_counts()
                flagged_count = int((counts > self.max_transactions_per_day).sum())

            if flagged_count == 0:
                return []

            return [RegulatoryViolation(
                rule_name=self.name,
                domain=self.domain,
                severity="WARNING",
                column=self.transaction_id_column,
                offending_count=flagged_count,
                message=(
                    f"[FATF Rec. 20] {flagged_count} account(s) exceed the velocity "
                    f"threshold of {self.max_transactions_per_day} transactions per day. "
                    "May indicate structuring, layering, or automated fraud activity."
                ),
                remediation=(
                    "Escalate flagged accounts to AML compliance team for manual review. "
                    "File SAR if structuring pattern is confirmed. "
                    "Review transaction limits and alert thresholds in the core banking system."
                ),
            )]
        except Exception as exc:  # noqa: BLE001
            logger.warning("SuspiciousTransactionPatternRule evaluation error: %s", exc)
            return []


class CurrencyConcentrationRule(BaseRegulatoryRule):
    """
    BCBS 239 — Risk data aggregation: concentration risk by currency.

    If a single currency dominates > max_concentration_pct of the dataset,
    the portfolio has excessive currency concentration risk, which reduces
    the reliability of risk aggregation reports.

    Regulatory basis: BCBS 239 Principle 6 (Adaptability), Basel III Pillar 2.
    """
    name = "currency_concentration"
    domain = "banking"

    def __init__(
        self,
        currency_column: str = "currency",
        max_concentration_pct: float = 0.90,
    ) -> None:
        self.currency_column = currency_column
        self.max_concentration_pct = max_concentration_pct

    def evaluate(self, df: pd.DataFrame) -> List[RegulatoryViolation]:
        if self._col_missing(self.currency_column, df):
            return []

        if df[self.currency_column].isna().all():
            return []

        counts = df[self.currency_column].dropna().value_counts(normalize=True)
        if counts.empty:
            return []

        top_currency = counts.index[0]
        top_pct = float(counts.iloc[0])

        if top_pct <= self.max_concentration_pct:
            return []

        return [RegulatoryViolation(
            rule_name=self.name,
            domain=self.domain,
            severity="WARNING",
            column=self.currency_column,
            offending_count=int((df[self.currency_column] == top_currency).sum()),
            message=(
                f"[BCBS239] Currency '{top_currency}' accounts for {top_pct:.1%} of "
                f"transactions, exceeding the concentration limit of "
                f"{self.max_concentration_pct:.0%}. "
                "High currency concentration reduces risk aggregation reliability."
            ),
            remediation=(
                "Diversify the dataset across multiple currencies, or ensure the "
                "concentration is intentional and documented in the risk framework. "
                "Update BCBS 239 risk data aggregation reports accordingly."
            ),
        )]
