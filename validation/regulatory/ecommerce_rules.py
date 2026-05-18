"""
validation/regulatory/ecommerce_rules.py
------------------------------------------
E-commerce and digital payments compliance rules.

Regulations covered
--------------------
PSD2 (EU 2015/2366)            — Strong Customer Authentication (SCA), refund rights
EU Consumer Rights Directive   — 14-day withdrawal right, pre-contract information
GDPR + ePrivacy                — Cookie consent, tracking transparency
EU P2B Regulation (2019/1150)  — Platform-to-Business fairness
Card Scheme Rules (Visa/MC)    — Chargeback threshold (< 1% of transactions)

Rules
-----
SCAExemptionTagRule             : PSD2 SCA — high-value transactions must have SCA flag
RefundRightWindowRule           : Consumer Rights Directive — refunds within 14 days
ChargebackThresholdRule         : Card scheme chargeback ratio < 1% (Visa CBP limit)
CookieConsentTrackingRule       : GDPR/ePrivacy — consent field required for tracked users
P2BRankingTransparencyRule      : EU P2B — ranking/listing criteria must be documented
ReturnReasonCaptureRule         : Return reason must be captured for all return records
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.ecommerce")

_PSD2_SCA_THRESHOLD = 30.0      # EUR — SCA required for transactions > €30
_CHARGEBACK_MAX_RATIO = 0.01    # 1% Visa CBP / MC excessive chargeback programme
_REFUND_WINDOW_DAYS = 14        # EU 14-day statutory withdrawal right


class SCAExemptionTagRule:
    """
    PSD2 Art. 97: Strong Customer Authentication (SCA) required for online payments > €30.
    Verifies that high-value transactions carry an SCA status field.
    """
    name = "PSD2_SCA_FLAG_REQUIRED"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        threshold = config.get("ecommerce", {}).get("sca_threshold_eur", _PSD2_SCA_THRESHOLD)
        amount_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"order_amount", "transaction_amount", "payment_amount", "basket_total"})),
            None
        )
        if not amount_col:
            return violations

        sca_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"sca_status", "sca_authenticated", "sca_exemption", "3ds_status"})),
            None
        )
        if sca_col:
            # Check that high-value transactions have SCA populated
            try:
                amounts = pd.to_numeric(df[amount_col], errors="coerce")
                high_value_mask = amounts > threshold
                missing_sca = high_value_mask & df[sca_col].isna()
                n_missing = int(missing_sca.sum())
                if n_missing > 0:
                    violations.append({
                        "rule": self.name, "severity": self.severity,
                        "column": sca_col,
                        "message": f"{n_missing} transactions > €{threshold:.0f} have missing SCA status. "
                                   "PSD2 Art. 97 requires SCA for high-value payments.",
                        "what_it_means": f"{n_missing} high-value transactions lack proof of Strong Customer Authentication.",
                        "why_it_matters": "Missing SCA evidence exposes PSP to chargebacks and PSD2 non-compliance liability.",
                        "recommended_action": "Ensure payment gateway records SCA outcome (3DS status) for all transactions > €30.",
                        "affected_rows": n_missing,
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("SCAExemptionTagRule error: %s", exc)
        else:
            # No SCA field at all — check if high-value transactions exist
            try:
                amounts = pd.to_numeric(df[amount_col], errors="coerce")
                high_value = (amounts > threshold).sum()
                if high_value > 0:
                    violations.append({
                        "rule": self.name, "severity": self.severity,
                        "column": "missing:sca_status",
                        "message": f"{high_value} transactions > €{threshold:.0f} detected but no SCA field found. "
                                   "PSD2 Art. 97 requires SCA authentication tracking.",
                        "what_it_means": "High-value online transactions lack SCA status tracking.",
                        "why_it_matters": "PSP liability for fraud chargebacks reverts to merchant without SCA evidence.",
                        "recommended_action": "Add 'sca_status' field from payment gateway (3DS passback data).",
                        "affected_rows": int(high_value),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("SCAExemptionTagRule error: %s", exc)
        return violations


class RefundRightWindowRule:
    """
    EU Consumer Rights Directive 2011/83/EU Art. 9:
    Consumers have 14 days to withdraw from distance contracts.
    Checks that refunds/returns are processed within this window.
    """
    name = "ECOMMERCE_REFUND_RIGHT_WINDOW"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        max_days = config.get("ecommerce", {}).get("refund_window_days", _REFUND_WINDOW_DAYS)
        refund_age_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"days_to_refund", "refund_processing_days", "return_days", "days_since_order"})),
            None
        )
        if not refund_age_col:
            return violations
        try:
            ages = pd.to_numeric(df[refund_age_col], errors="coerce").dropna()
            late = (ages > max_days).sum()
            if late > 0:
                violations.append({
                    "rule": self.name, "severity": self.severity,
                    "column": refund_age_col,
                    "message": f"{late} refund/return(s) processed after the {max_days}-day statutory window.",
                    "what_it_means": f"{late} consumers may have been denied their statutory right to a refund within {max_days} days.",
                    "why_it_matters": "EU CRD Art. 9 statutory right — violations reportable to national Trading Standards.",
                    "recommended_action": "Implement SLA alerts for refund requests approaching the 14-day window.",
                    "affected_rows": int(late),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("RefundRightWindowRule error: %s", exc)
        return violations


class ChargebackThresholdRule:
    """
    Visa CBP / Mastercard ECP: Chargeback ratio > 1% triggers programme monitoring.
    Ratio = Chargebacks / Total Transactions (count or value).
    """
    name = "ECOMMERCE_CHARGEBACK_THRESHOLD"
    severity = "CRITICAL"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        max_ratio = config.get("ecommerce", {}).get("chargeback_max_ratio", _CHARGEBACK_MAX_RATIO)
        cb_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"is_chargeback", "chargeback_flag", "disputed", "dispute_flag"})),
            None
        )
        if not cb_col:
            return violations
        try:
            total = len(df[cb_col].dropna())
            chargebacks = int(df[cb_col].astype(str).str.lower().isin({"1", "true", "yes", "disputed"}).sum())
            if total == 0:
                return violations
            ratio = chargebacks / total
            if ratio > max_ratio:
                violations.append({
                    "rule": self.name, "severity": self.severity,
                    "column": cb_col,
                    "message": f"Chargeback ratio {ratio:.2%} exceeds {max_ratio:.1%} threshold "
                               f"({chargebacks:,} of {total:,} transactions). "
                               "Visa CBP / Mastercard ECP programme monitoring triggered.",
                    "what_it_means": f"1 in {int(1/ratio)} transactions results in a chargeback — above card scheme limits.",
                    "why_it_matters": "Merchants above 1% risk Excessive Chargeback Programme (increased fees/termination).",
                    "recommended_action": "Implement 3DS, fraud screening, and clear product descriptions to reduce chargebacks.",
                    "affected_rows": chargebacks,
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("ChargebackThresholdRule error: %s", exc)
        return violations


class CookieConsentTrackingRule:
    """
    GDPR Art. 6 + ePrivacy Directive:
    Any dataset with user tracking data must include a consent column.
    """
    name = "ECOMMERCE_COOKIE_CONSENT_TRACKING"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        tracking_hints = {"session_id", "click", "page_view", "device_id", "ip_address",
                          "user_agent", "tracker", "analytics_id", "cookie"}
        has_tracking = any(any(h in c.lower() for h in tracking_hints) for c in df.columns)
        if not has_tracking:
            return violations

        consent_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"cookie_consent", "tracking_consent", "analytics_consent", "gdpr_consent"})),
            None
        )
        if not consent_col:
            violations.append({
                "rule": self.name, "severity": self.severity,
                "column": "missing:cookie_consent",
                "message": "User tracking data detected but no cookie/tracking consent column found. "
                           "GDPR Art. 6 + ePrivacy Directive require consent for non-essential tracking.",
                "what_it_means": "User behavioural tracking without consent evidence violates GDPR.",
                "why_it_matters": "ICO/CNIL fines for cookie consent violations up to 4% of global revenue.",
                "recommended_action": "Implement Consent Management Platform (CMP). Add 'cookie_consent' field to user tracking events.",
                "affected_rows": len(df),
            })
        return violations


class P2BRankingTransparencyRule:
    """
    EU Platform-to-Business Regulation (2019/1150) Art. 5:
    Online platforms must document the main parameters determining ranking.
    Checks for ranking_score field in product/seller listings.
    """
    name = "ECOMMERCE_P2B_RANKING_TRANSPARENCY"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        listing_hints = {"product_id", "sku", "seller_id", "listing_id", "search_result"}
        is_listing = any(any(h in c.lower() for h in listing_hints) for c in df.columns)
        if not is_listing:
            return violations

        # Check ranking position field
        rank_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"rank", "position", "search_rank", "listing_rank", "relevance_score"})),
            None
        )
        if rank_col:
            # Good — ranking exists; check if score/factor columns present
            score_col = next(
                (c for c in df.columns if any(h in c.lower() for h in
                 {"ranking_factor", "ranking_score", "relevance_factor", "algorithm_score"})),
                None
            )
            if not score_col:
                violations.append({
                    "rule": self.name, "severity": self.severity,
                    "column": rank_col,
                    "message": "Listing rank column found but no ranking factor/score column present. "
                               "EU P2B Art. 5 requires disclosure of main ranking parameters.",
                    "what_it_means": "Ranking logic exists but the contributing factors are not captured in the data.",
                    "why_it_matters": "Sellers on your platform have the right to understand ranking factors under EU P2B.",
                    "recommended_action": "Add 'ranking_score_factors' JSON column documenting top ranking signals per listing.",
                    "affected_rows": 0,
                })
        return violations


class ReturnReasonCaptureRule:
    """
    Returning goods without capturing the reason prevents consumer rights analysis.
    Verifies return records include a return_reason field.
    """
    name = "ECOMMERCE_RETURN_REASON_CAPTURE"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        return_hints = {"return_id", "returned", "is_return", "return_flag", "return_status"}
        has_returns = any(any(h in c.lower() for h in return_hints) for c in df.columns)
        if not has_returns:
            return violations

        reason_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"return_reason", "reason_for_return", "refund_reason", "return_code"})),
            None
        )
        if not reason_col:
            violations.append({
                "rule": self.name, "severity": self.severity,
                "column": "missing:return_reason",
                "message": "Return records detected but no return_reason column found.",
                "what_it_means": "Returns are not classified by reason — prevents consumer rights and returns analytics.",
                "why_it_matters": "DEFRA, consumer rights regulators, and card schemes require reason tracking for dispute resolution.",
                "recommended_action": "Add 'return_reason' with standard codes: DAMAGED, NOT_AS_DESCRIBED, CHANGED_MIND, FAULTY, OTHER.",
                "affected_rows": 0,
            })
        else:
            # Check for null reasons on return records
            try:
                return_col = next(
                    c for c in df.columns if any(h in c.lower() for h in return_hints)
                )
                is_return_mask = df[return_col].astype(str).str.lower().isin(
                    {"1", "true", "yes", "returned", "return"}
                )
                missing_reason = (is_return_mask & df[reason_col].isna()).sum()
                if missing_reason > 0:
                    violations.append({
                        "rule": self.name, "severity": self.severity,
                        "column": reason_col,
                        "message": f"{missing_reason} return record(s) have no return reason captured.",
                        "what_it_means": f"{missing_reason} returned items cannot be categorised for consumer rights analysis.",
                        "why_it_matters": "Missing reasons prevent trend analysis needed to reduce preventable returns.",
                        "recommended_action": "Enforce return_reason field at point of return initiation in the customer journey.",
                        "affected_rows": int(missing_reason),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("ReturnReasonCaptureRule error: %s", exc)
        return violations
