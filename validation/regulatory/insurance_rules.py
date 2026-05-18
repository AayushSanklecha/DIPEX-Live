"""
validation/regulatory/insurance_rules.py
------------------------------------------
Insurance-specific regulatory compliance rules.

Regulations covered
--------------------
Solvency II (EU 2009/138/EC)   — SCR ratio, ORSA requirements
IFRS 17                        — contractual service margin (CSM) tracking
Lloyd's/FCA ICOBs              — claims settlement timeliness
IAIS ICP 13                    — reinsurance exposure limits

Rules
-----
SCRCapitalAdequacyRule         : Solvency Capital Requirement ratio ≥ 100%
CombinedRatioRule              : Combined ratio (loss + expense) sanity check (< 115%)
ClaimSettlementTimelinessRule  : Claim age field must be within thresholds
ReinsuranceExposureRule        : Reinsurance ceded ratio within regulatory limits
PremiumIntegrityRule           : Premium must be positive and non-zero for active policies
CSMTrackingRule                : IFRS 17 CSM field presence and sign validation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.insurance")

_SCR_MIN_RATIO = 1.0           # 100% SCR coverage required (Solvency II)
_COMBINED_RATIO_WARN = 1.0     # > 100% → underwriting loss
_COMBINED_RATIO_FAIL = 1.15    # > 115% → critical
_MAX_REINSURANCE_CESSION = 0.9 # > 90% ceded → concentration risk


class SCRCapitalAdequacyRule:
    """
    Solvency Capital Requirement ratio = Eligible Own Funds / SCR ≥ 100%.
    Solvency II Art. 100 — firms below 100% SCR must notify regulator immediately.
    """
    name = "INSURANCE_SCR_CAPITAL_ADEQUACY"
    severity = "CRITICAL"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        eof_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"eligible_own_funds", "eof", "own_funds", "capital"})),
            None
        )
        scr_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"scr", "solvency_capital", "capital_requirement"})),
            None
        )
        if not (eof_col and scr_col):
            return violations
        try:
            eof  = pd.to_numeric(df[eof_col], errors="coerce")
            scr  = pd.to_numeric(df[scr_col], errors="coerce")
            ratio = (eof / scr.replace(0, float("nan"))).dropna()
            breaches = (ratio < _SCR_MIN_RATIO).sum()
            if breaches > 0:
                min_ratio = float(ratio.min())
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": f"{eof_col}, {scr_col}",
                    "message": f"{breaches} record(s) have SCR ratio below 100% (min={min_ratio:.1%}). "
                               "Solvency II Art. 100 mandates ≥100% SCR coverage.",
                    "what_it_means": "The insurer has insufficient capital relative to its risk-based capital requirement.",
                    "why_it_matters": "EIOPA must be notified immediately; firm faces potential withdrawal of authorisation.",
                    "recommended_action": "Immediately assess capital position, trigger ORSA, and notify supervisory authority.",
                    "affected_rows": int(breaches),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("SCRCapitalAdequacyRule error: %s", exc)
        return violations


class CombinedRatioRule:
    """
    Combined ratio = (Incurred Losses + LAE + Underwriting Expenses) / Earned Premium.
    > 100% = underwriting loss; > 115% = critical; industry benchmark = 95-100%.
    """
    name = "INSURANCE_COMBINED_RATIO"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        cr_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"combined_ratio", "loss_ratio", "combined_loss"})),
            None
        )
        if not cr_col:
            # Try to compute from components
            loss_col = next((c for c in df.columns if "incurred_loss" in c.lower() or "loss_amount" in c.lower()), None)
            prem_col = next((c for c in df.columns if "earned_premium" in c.lower() or "gross_premium" in c.lower()), None)
            if loss_col and prem_col:
                try:
                    ratio = (pd.to_numeric(df[loss_col], errors="coerce") /
                             pd.to_numeric(df[prem_col], errors="coerce").replace(0, float("nan")))
                    cr_col = "_computed_cr"
                    df = df.copy()
                    df[cr_col] = ratio
                except Exception:
                    return violations
            else:
                return violations

        try:
            ratios = pd.to_numeric(df[cr_col], errors="coerce").dropna()
            critical = (ratios > _COMBINED_RATIO_FAIL).sum()
            warn = ((ratios > _COMBINED_RATIO_WARN) & (ratios <= _COMBINED_RATIO_FAIL)).sum()
            if critical > 0:
                violations.append({
                    "rule": self.name, "severity": "CRITICAL",
                    "column": cr_col,
                    "message": f"{critical} record(s) have combined ratio > 115% ({ratios.max():.1%}). "
                               "This indicates severe underwriting losses.",
                    "what_it_means": "The insurer is paying out significantly more in claims than it earns in premiums.",
                    "why_it_matters": "Sustained combined ratios >115% lead to insolvency. Requires ORSA triggers.",
                    "recommended_action": "Review pricing strategy, claims reserves, and expense management. Trigger ORSA review.",
                    "affected_rows": int(critical),
                })
            elif warn > 0:
                violations.append({
                    "rule": self.name, "severity": "WARNING",
                    "column": cr_col,
                    "message": f"{warn} record(s) have combined ratio > 100% (underwriting loss territory).",
                    "what_it_means": "Underwriting losses detected — more paid in claims/expenses than earned in premiums.",
                    "why_it_matters": "Sustained underwriting losses erode capital; must be offset by investment income.",
                    "recommended_action": "Monitor investment income offset; review pricing and claims experience.",
                    "affected_rows": int(warn),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("CombinedRatioRule error: %s", exc)
        return violations


class ClaimSettlementTimelinessRule:
    """
    FCA ICOBS 8.1: Claims must be settled promptly.
    Flags claims open longer than 180 days (configurable).
    """
    name = "INSURANCE_CLAIM_SETTLEMENT_TIMELINESS"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        max_days = config.get("insurance", {}).get("max_claim_open_days", 180)
        age_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"claim_age", "days_open", "claim_duration", "days_since_fnol"})),
            None
        )
        if not age_col:
            return violations
        try:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            stale = (ages > max_days).sum()
            if stale > 0:
                violations.append({
                    "rule": self.name,
                    "severity": self.severity,
                    "column": age_col,
                    "message": f"{stale} claim(s) open longer than {max_days} days (max={ages.max():.0f} days). "
                               "FCA ICOBS 8.1 requires prompt settlement.",
                    "what_it_means": f"{stale} claims have been outstanding beyond the {max_days}-day threshold.",
                    "why_it_matters": "Delayed claim settlement breaches FCA conduct rules and exposes firm to complaints.",
                    "recommended_action": "Prioritise aged claims. Investigate bottlenecks in claims handling workflow.",
                    "affected_rows": int(stale),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("ClaimSettlementTimelinessRule error: %s", exc)
        return violations


class ReinsuranceExposureRule:
    """
    IAIS ICP 13: Excessive reinsurance cession concentration is a risk.
    Flags where ceded reinsurance > 90% of gross written premium.
    """
    name = "INSURANCE_REINSURANCE_EXPOSURE"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        ceded_col = next((c for c in df.columns if "ceded" in c.lower() or "reinsurance_premium" in c.lower()), None)
        gwp_col   = next((c for c in df.columns if any(h in c.lower() for h in
                          {"gross_written_premium", "gwp", "written_premium"})), None)
        if not (ceded_col and gwp_col):
            return violations
        try:
            ceded = pd.to_numeric(df[ceded_col], errors="coerce")
            gwp   = pd.to_numeric(df[gwp_col], errors="coerce")
            ratio = (ceded / gwp.replace(0, float("nan"))).dropna()
            high  = (ratio > _MAX_REINSURANCE_CESSION).sum()
            if high > 0:
                violations.append({
                    "rule": self.name, "severity": self.severity,
                    "column": f"{ceded_col}, {gwp_col}",
                    "message": f"{high} record(s) cede >90% of GWP to reinsurers. "
                               "IAIS ICP 13: excessive reinsurance creates counterparty concentration risk.",
                    "what_it_means": "The insurer is highly dependent on a small number of reinsurers.",
                    "why_it_matters": "Reinsurer insolvency could cascade losses back to the primary insurer.",
                    "recommended_action": "Diversify reinsurance panel. Review counterparty credit quality.",
                    "affected_rows": int(high),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("ReinsuranceExposureRule error: %s", exc)
        return violations


class PremiumIntegrityRule:
    """
    Premium must be positive for active policies.
    Zero or negative premiums indicate data errors or refund misclassification.
    """
    name = "INSURANCE_PREMIUM_INTEGRITY"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        prem_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"premium", "written_premium", "earned_premium", "gross_premium"})),
            None
        )
        if not prem_col:
            return violations
        try:
            premiums = pd.to_numeric(df[prem_col], errors="coerce").dropna()
            bad = (premiums <= 0).sum()
            if bad > 0:
                violations.append({
                    "rule": self.name, "severity": self.severity,
                    "column": prem_col,
                    "message": f"{bad} record(s) have zero or negative premium. "
                               "Active policies must have positive premium values.",
                    "what_it_means": "Premiums of ≤0 indicate data entry errors, refund misclassification, or reversed transactions.",
                    "why_it_matters": "Incorrect premium data distorts loss ratios, pricing models, and regulatory capital calculations.",
                    "recommended_action": "Audit premium entries. Move refunds/credits to a separate 'adjustments' table.",
                    "affected_rows": int(bad),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("PremiumIntegrityRule error: %s", exc)
        return violations


class CSMTrackingRule:
    """
    IFRS 17: Contractual Service Margin (CSM) must be non-negative at initial recognition.
    Negative CSM at inception = onerous contract — must be recognised immediately in P&L.
    """
    name = "INSURANCE_IFRS17_CSM_TRACKING"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        csm_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"csm", "contractual_service_margin", "ifrs17_csm"})),
            None
        )
        if not csm_col:
            # Flag if policy data present but no CSM column
            policy_hints = {"policy_id", "policy_number", "insurance_contract"}
            if any(any(h in c.lower() for h in policy_hints) for c in df.columns):
                violations.append({
                    "rule": self.name, "severity": "WARNING",
                    "column": "missing:csm",
                    "message": "Insurance contract data present but no CSM (Contractual Service Margin) column found. "
                               "IFRS 17 requires CSM tracking for all insurance contracts.",
                    "what_it_means": "IFRS 17 measurement model requires tracking deferred profit in the CSM.",
                    "why_it_matters": "Missing CSM tracking means IFRS 17 compliance cannot be verified.",
                    "recommended_action": "Add 'csm' column and calculate Contractual Service Margin at contract inception.",
                    "affected_rows": 0,
                })
            return violations

        try:
            csm = pd.to_numeric(df[csm_col], errors="coerce").dropna()
            initial_negative = (csm < 0).sum()
            if initial_negative > 0:
                violations.append({
                    "rule": self.name, "severity": self.severity,
                    "column": csm_col,
                    "message": f"{initial_negative} record(s) have negative CSM — these contracts are onerous under IFRS 17.",
                    "what_it_means": "Contracts with negative CSM at inception must recognise losses immediately in P&L.",
                    "why_it_matters": "Onerous contracts require immediate loss recognition — not doing so misstates earnings.",
                    "recommended_action": "Re-classify these contracts as onerous; recognise loss component in P&L immediately.",
                    "affected_rows": int(initial_negative),
                })
        except Exception as exc:  # noqa: BLE001
            logger.debug("CSMTrackingRule error: %s", exc)
        return violations
