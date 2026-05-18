"""
validation/regulatory/mifid2_rules.py
---------------------------------------
MiFID II compliance rules (EU Markets in Financial Instruments Directive II).

Rules
-----
BestExecutionRule          : Prices within 2% of VWAP reference (Art. 27)
TransactionReportingRule   : Required fields: LEI, ISIN, venue code
AlgoOrderTagRule           : Algorithmic orders must have algo_id tag (RTS 6)
InvestorCategorizationRule : Client category must be valid (retail/professional/eligible-counterparty)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.mifid2")

_VALID_INVESTOR_CATEGORIES = {"retail", "professional", "eligible_counterparty",
                               "eligible-counterparty", "prof", "retail_client"}
_VENUE_CODES = {"XLON", "XPAR", "XFRA", "XAMS", "XMIL", "XBRU", "XDUB", "XIST",
                "XNAS", "XNYS", "XCHI", "BATS", "CBOE"}


class BestExecutionRule:
    """
    MiFID II Art. 27: Execution venues must achieve best execution.
    Verifies trade prices are within 2% of VWAP reference price.
    """
    name = "MIFID2_BEST_EXECUTION"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        price_col = next(
            (c for c in df.columns if any(h in c.lower() for h in {"price", "trade_price", "exec_price"})),
            None
        )
        vwap_col = next(
            (c for c in df.columns if "vwap" in c.lower()),
            None
        )
        if price_col and vwap_col:
            try:
                prices = pd.to_numeric(df[price_col], errors="coerce")
                vwap = pd.to_numeric(df[vwap_col], errors="coerce")
                deviation = ((prices - vwap).abs() / vwap.replace(0, float("nan"))).dropna()
                breaches = (deviation > 0.02).sum()
                if breaches > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": f"{price_col}, {vwap_col}",
                        "message": f"{breaches} trade(s) deviate >2% from VWAP reference. "
                                   "MiFID II Art. 27 best execution obligation may be breached.",
                        "what_it_means": f"{breaches} transactions were executed at prices >2% away from the market average.",
                        "why_it_matters": "Best execution failure exposes the firm to regulatory fines and client litigation.",
                        "recommended_action": "Investigate execution venues and routing logic for these trades.",
                        "affected_rows": int(breaches),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("BestExecutionRule error: %s", exc)
        return violations


class TransactionReportingRule:
    """
    MiFID II RTS 22: Requires LEI, ISIN, and venue code fields in transaction reports.
    """
    name = "MIFID2_TRANSACTION_REPORTING"
    severity = "ERROR"

    REQUIRED_FIELD_HINTS = {
        "lei": ["lei", "legal_entity_id"],
        "isin": ["isin", "instrument_id"],
        "venue": ["venue_code", "mic_code", "trading_venue", "execution_venue"],
    }

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        cols_lower = {c.lower(): c for c in df.columns}
        for field_name, hints in self.REQUIRED_FIELD_HINTS.items():
            found = any(
                any(h in c for c in cols_lower)
                for h in hints
            )
            if not found:
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": f"missing:{field_name}",
                    "message": f"Required MiFID II transaction reporting field '{field_name}' not found. "
                               "RTS 22 mandates LEI, ISIN, and venue code.",
                    "what_it_means": f"The dataset is missing the '{field_name}' column required for regulatory reporting.",
                    "why_it_matters": "Incomplete transaction reports can trigger FCA/ESMA enforcement action.",
                    "recommended_action": f"Add the '{field_name}' field to transaction records before reporting.",
                    "affected_rows": len(df),
                })
        return violations


class AlgoOrderTagRule:
    """
    MiFID II RTS 6: Algorithmic orders must carry an algo_id tag.
    """
    name = "MIFID2_ALGO_ORDER_TAG"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        order_type_col = next(
            (c for c in df.columns if "order_type" in c.lower() or "order_flag" in c.lower()),
            None
        )
        algo_col = next(
            (c for c in df.columns if "algo_id" in c.lower() or "algo_flag" in c.lower()),
            None
        )
        if order_type_col and not algo_col:
            algo_rows = df[order_type_col].astype(str).str.lower().str.contains("algo|automated|program", na=False).sum()
            if algo_rows > 0:
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": order_type_col,
                    "message": f"{algo_rows} algorithmic order(s) detected but no algo_id column found. "
                               "MiFID II RTS 6 requires algo order identification.",
                    "what_it_means": "Algorithmic orders cannot be identified in regulatory audits.",
                    "why_it_matters": "RTS 6 non-compliance prevents proper algo trading surveillance.",
                    "recommended_action": "Add 'algo_id' column to tag all algorithmic orders with a unique identifier.",
                    "affected_rows": int(algo_rows),
                })
        return violations


class InvestorCategorizationRule:
    """
    MiFID II: Client category field must contain valid categories.
    """
    name = "MIFID2_INVESTOR_CATEGORIZATION"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        cat_col = next(
            (c for c in df.columns if any(h in c.lower()
             for h in {"client_category", "investor_type", "client_type", "customer_category"})),
            None
        )
        if cat_col:
            try:
                values = df[cat_col].dropna().astype(str).str.lower().str.strip()
                invalid = values[~values.isin(_VALID_INVESTOR_CATEGORIES)]
                if len(invalid) > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": cat_col,
                        "message": f"{len(invalid)} row(s) in '{cat_col}' have invalid investor categories: "
                                   f"{list(invalid.unique()[:5])}.",
                        "what_it_means": "Client categorization is incorrect, affecting product suitability assessments.",
                        "why_it_matters": "Mis-categorized clients may receive unsuitable products, violating MiFID II.",
                        "recommended_action": f"Standardize to: {sorted(_VALID_INVESTOR_CATEGORIES)}.",
                        "affected_rows": len(invalid),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("InvestorCategorizationRule error: %s", exc)
        return violations
