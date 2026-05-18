"""
validation/regulatory/esg_rules.py
-------------------------------------
ESG (Environmental, Social, Governance) compliance rules.

Rules
-----
CarbonEmissionsRule           : Scope 1/2/3 emissions must be non-negative and plausible
ESGScoreRangeRule             : ESG score columns must be within published rating scale
GenderPayGapDisclosureRule    : Salary/compensation columns must allow gender grouping
SupplyChainDueDiligenceRule   : Flags missing supplier risk ratings in supply chain data
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("dipex.validation.regulatory.esg")

_ESG_SCORE_RANGES = {
    "msci": (0, 100),
    "sustainalytics": (0, 100),
    "ftse": (0, 5),
    "bloomberg": (0, 100),
    "default": (0, 100),
}


class CarbonEmissionsRule:
    """
    Validate Scope 1, 2, and 3 emissions columns:
    - Must be non-negative
    - Detect implausible spikes (>10x median)
    Aligned with GHG Protocol and TCFD recommendations.
    """
    name = "ESG_CARBON_EMISSIONS"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        scope_hints = ["scope_1", "scope_2", "scope_3", "scope1", "scope2", "scope3",
                       "co2_emissions", "ghg_emissions", "carbon_footprint"]
        for col in df.columns:
            if not any(h in col.lower() for h in scope_hints):
                continue
            try:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(values) == 0:
                    continue
                negatives = (values < 0).sum()
                if negatives > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": "ERROR",
                        "column": col,
                        "message": f"Column '{col}' has {negatives} negative emission value(s). "
                                   "GHG emissions cannot be negative.",
                        "what_it_means": "Negative carbon emissions values indicate data errors or incorrect sign convention.",
                        "why_it_matters": "Incorrect emissions data will fail GHG Protocol verification and TCFD reporting.",
                        "recommended_action": "Verify data source. Convert absolute reductions to credits (separate field).",
                        "affected_rows": int(negatives),
                    })
                median = values.median()
                spikes = (values > 10 * median).sum()
                if spikes > 0 and median > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": col,
                        "message": f"Column '{col}' has {spikes} spike(s) >10× median "
                                   f"(median={median:,.1f}). Possible data error.",
                        "what_it_means": "Emissions values are implausibly high compared to the typical range.",
                        "why_it_matters": "Emission spikes will distort ESG scores and fail sustainability audits.",
                        "recommended_action": "Investigate spike rows, check measurement units, verify data pipeline.",
                        "affected_rows": int(spikes),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("CarbonEmissionsRule error on col '%s': %s", col, exc)
        return violations


class ESGScoreRangeRule:
    """
    Verify ESG score columns are within the expected rating scale (typically 0–100 or 0–10).
    """
    name = "ESG_SCORE_RANGE"
    severity = "ERROR"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        score_hints = ["esg_score", "sustainability_score", "esg_rating", "environmental_score",
                       "social_score", "governance_score", "esg_grade"]
        for col in df.columns:
            if not any(h in col.lower() for h in score_hints):
                continue
            try:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(values) == 0:
                    continue
                # Auto-detect scale
                max_val = values.max()
                if max_val <= 10:
                    lo, hi = 0, 10
                elif max_val <= 100:
                    lo, hi = 0, 100
                else:
                    lo, hi = 0, max_val  # unknown scale

                out_of_range = ((values < lo) | (values > hi)).sum()
                if out_of_range > 0:
                    violations.append({
                        "rule": self.name,
                        "severity": self.severity,
                        "column": col,
                        "message": f"Column '{col}': {out_of_range} ESG score(s) outside expected range "
                                   f"[{lo}, {hi}].",
                        "what_it_means": "ESG ratings contain values outside the standard scale — likely data entry errors.",
                        "why_it_matters": "Invalid ESG scores cannot be used for screening, fund allocation, or disclosure.",
                        "recommended_action": f"Clamp or re-source ESG scores to [{lo}, {hi}] range.",
                        "affected_rows": int(out_of_range),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.debug("ESGScoreRangeRule error on col '%s': %s", col, exc)
        return violations


class GenderPayGapDisclosureRule:
    """
    Check if salary/compensation data can be disaggregated by gender.
    Required under EU Pay Transparency Directive (2023/970) and UK Gender Pay Gap reporting.
    """
    name = "ESG_GENDER_PAY_GAP_DISCLOSURE"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        salary_hints = ["salary", "compensation", "pay", "wage", "remuneration", "earnings"]
        has_salary = any(any(h in c.lower() for h in salary_hints) for c in df.columns)
        if not has_salary:
            return violations

        gender_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"gender", "sex", "gender_identity"})),
            None
        )
        if not gender_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": "missing:gender",
                "message": "Salary/compensation data present but no gender column found. "
                           "EU Pay Transparency Directive requires pay gap disclosure by gender.",
                "what_it_means": "Gender pay gap cannot be calculated or disclosed without a gender identifier.",
                "why_it_matters": "EU firms with >100 employees must report gender pay gaps; failure risks fines.",
                "recommended_action": "Add 'gender' or 'gender_identity' column with appropriate consent handling.",
                "affected_rows": len(df),
            })
        return violations


class SupplyChainDueDiligenceRule:
    """
    Flag missing supplier risk ratings when supply chain data is present.
    Required under EU Supply Chain Act (CSDDD) and German LkSG.
    """
    name = "ESG_SUPPLY_CHAIN_DUE_DILIGENCE"
    severity = "WARNING"

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
        violations = []
        supply_hints = ["supplier", "vendor", "manufacturer", "supply_chain", "tier_1", "tier_2"]
        has_supply = any(any(h in c.lower() for h in supply_hints) for c in df.columns)
        if not has_supply:
            return violations

        risk_col = next(
            (c for c in df.columns if any(h in c.lower() for h in
             {"supplier_risk", "vendor_risk", "esg_risk_rating", "due_diligence_score"})),
            None
        )
        if not risk_col:
            violations.append({
                "rule": self.name,
                "severity": self.severity,
                "column": "missing:supplier_risk_rating",
                "message": "Supply chain data detected but no supplier risk rating column found. "
                           "EU CSDDD and German LkSG require supplier ESG due diligence.",
                "what_it_means": "No supplier risk assessment data attached — cannot verify ethical sourcing.",
                "why_it_matters": "CSDDD non-compliance can result in liability for human rights violations in supply chains.",
                "recommended_action": "Add 'supplier_risk_rating' and conduct ESG due diligence assessments.",
                "affected_rows": len(df),
            })
        return violations
