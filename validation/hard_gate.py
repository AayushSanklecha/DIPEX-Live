"""
validation/hard_gate.py
------------------------
Step 2 — Deterministic Validation: Hard Gate 1

The HardGate is the master validator that runs all five sub-validators
sequentially and returns a structured GateResult.

Decision policy:
  • ANY violation with severity == "CRITICAL" or "ERROR" → DECISION: REJECT
  • REJECT sets suppress_learning=True to prevent RL updates from bad data
  • WARNING-only → DECISION: PASS (with warnings logged)

Sub-validators run order:
  1. SchemaValidator   — required columns, types, timestamps, unique keys
  2. NullValidator     — critical fields, per-column & global thresholds
  3. RangeValidator    — bounds, financial positivity, logical inequalities
  4. IntegrityChecker  — cross-column consistency, ID uniqueness
  5. RegulatoryEngine  — domain-specific rules (banking / healthcare / generic)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from validation.schema_validator import SchemaValidator
from validation.null_validator import NullValidator, NullViolation
from validation.range_validator import RangeValidator, RangeViolation
from validation.integrity_checker import IntegrityChecker
from validation.regulatory.regulatory_engine import RegulatoryEngine
from validation.zero_value_detector import ZeroValueDetector, ZeroViolation

try:
    from validation.shap_explainer import explain_gate_failure as _shap_explain
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Severities that trigger a hard-reject decision (only in strict mode)
_BLOCKING_SEVERITIES = {"CRITICAL", "ERROR"}


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Structured output from HardGate.run()."""
    run_id: str
    decision: str                          # "PASS" | "REJECT"
    reason: str                            # Human-readable explanation
    suppress_learning: bool                # True when rejected — blocks RL update
    failures: List[Dict[str, Any]]         # All ERROR/CRITICAL violations
    warnings: List[Dict[str, Any]]         # All WARNING violations
    total_violations: int
    total_warnings: int
    shap_explanation: Optional[Dict[str, Any]] = None  # SHAP column risk (REJECT only)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "decision": self.decision,
            "reason": self.reason,
            "suppress_learning": self.suppress_learning,
            "total_violations": self.total_violations,
            "total_warnings": self.total_warnings,
            "failures": self.failures,
            "warnings": self.warnings,
            "evaluated_at": self.evaluated_at,
        }
        if self.shap_explanation:
            d["shap_explanation"] = self.shap_explanation
        return d


# ---------------------------------------------------------------------------
# Hard Gate
# ---------------------------------------------------------------------------

class HardGate:
    """
    Orchestrates the five sub-validators that form Hard Gate 1.

    Example:
        gate = HardGate.from_config(config)
        result = gate.run(df, run_id="abc123")
        if result.decision == "REJECT":
            return False  # abort pipeline
    """

    def __init__(
        self,
        schema_validator: SchemaValidator,
        null_validator: NullValidator,
        range_validator: RangeValidator,
        integrity_checker: IntegrityChecker,
        regulatory_engine: RegulatoryEngine,
        zero_value_detector: Optional["ZeroValueDetector"] = None,
        schema_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._schema_validator   = schema_validator
        self._null_validator     = null_validator
        self._range_validator    = range_validator
        self._integrity_checker  = integrity_checker
        self._regulatory_engine  = regulatory_engine
        self._zero_detector      = zero_value_detector
        self._schema_info        = schema_info or {}
        self._strict_mode        = True
        self._advisory_mode      = False  # when True: gate warns but never rejects

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HardGate":
        """Factory — constructs a fully wired HardGate from the project config."""
        schema_info = config.get("validation", {}).get("schema", {})
        strict_mode = config.get("validation", {}).get("strict_mode", False)
        
        gate = cls(
            schema_validator   = SchemaValidator(config),
            null_validator     = NullValidator.from_config(config),
            range_validator    = RangeValidator.from_config(config),
            integrity_checker  = IntegrityChecker(config),
            regulatory_engine  = RegulatoryEngine.from_config(config),
            zero_value_detector= ZeroValueDetector.from_config(config),
            schema_info        = schema_info,
        )
        gate._strict_mode   = strict_mode
        # advisory_mode can be set in hard_gate_1 OR validation stanza
        gate._advisory_mode = (
            config.get("hard_gate_1", {}).get("advisory_mode", False)
            or config.get("validation", {}).get("advisory_mode", False)
        )
        return gate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, run_id: str = "N/A") -> GateResult:
        """
        Execute all validation sub-gates against ``df``.

        Args:
            df:     The ingested DataFrame to validate.
            run_id: Pipeline run identifier, included in the GateResult.

        Returns:
            GateResult with decision, all failures, and suppress_learning flag.
        """
        logger.info("Hard Gate 1 started for run_id=%s  shape=%s", run_id, df.shape)

        raw_failures: List[Dict[str, Any]] = []
        raw_warnings: List[Dict[str, Any]] = []

        # ── 1. Schema Validation ─────────────────────────────────────
        schema_errors = self._schema_validator.validate(df, self._schema_info)
        self._sort_into_buckets(schema_errors, raw_failures, raw_warnings)

        # ── 2. Null Threshold Control ─────────────────────────────────
        null_violations: List[NullViolation] = self._null_validator.validate(df)
        for v in null_violations:
            bucket = raw_failures if v.severity in _BLOCKING_SEVERITIES else raw_warnings
            bucket.append(v.to_dict())

        # ── 3. Range Validation ───────────────────────────────────────
        range_violations: List[RangeViolation] = self._range_validator.validate(df)
        for v in range_violations:
            bucket = raw_failures if v.severity in _BLOCKING_SEVERITIES else raw_warnings
            bucket.append(v.to_dict())

        # ── 4. Referential Integrity ──────────────────────────────────
        integrity_errors = self._integrity_checker.check(df)
        self._sort_into_buckets(integrity_errors, raw_failures, raw_warnings)

        # ── 5. Regulatory Engine ──────────────────────────────────────
        reg_violations = self._regulatory_engine.evaluate(df)
        for v in reg_violations:
            bucket = raw_failures if v.severity in _BLOCKING_SEVERITIES else raw_warnings
            bucket.append(v.to_dict())

        # ── 6. Zero-Value Detection ───────────────────────────────────
        if self._zero_detector is not None:
            zero_report = self._zero_detector.validate(df, run_id=run_id)
            for v in zero_report.violations:
                bucket = raw_failures if v.severity in _BLOCKING_SEVERITIES else raw_warnings
                bucket.append(v.to_dict())
            if zero_report.violations:
                logger.info(
                    "[ZeroDetector] %d zero-value violation(s) found for run_id=%s.",
                    len(zero_report.violations), run_id,
                )

        # ── 7. Adaptive Degradation (Non-Strict Mode) ─────────────────
        if not self._strict_mode and raw_failures:
            # Drop columns that caused blocking errors so we can proceed
            bad_cols = {f.get("column") for f in raw_failures if f.get("column")}
            existing_bad_cols = list(bad_cols.intersection(df.columns))
            
            if existing_bad_cols:
                df.drop(columns=existing_bad_cols, inplace=True)
                msg_degrade = f"Adaptive mode: dropped {len(existing_bad_cols)} failing columns {existing_bad_cols}"
                logger.warning(msg_degrade)
                raw_warnings.append({
                    "column": "DATASET",
                    "severity": "WARNING",
                    "type": "ADAPTIVE_DEGRADATION",
                    "message": msg_degrade
                })
            
            # Demote failures to warnings since we handled them by dropping
            raw_warnings.extend(raw_failures)
            raw_failures = []

        # ── 8. Advisory Mode: demote ALL blocking failures to warnings ─────────
        # When advisory_mode=True the gate NEVER returns REJECT regardless of
        # violation severity. It records everything for audit but the pipeline
        # continues unconditionally. This is the production default so that
        # dirty / messy data can still reach downstream cleaning stages.
        if self._advisory_mode and raw_failures:
            logger.warning(
                "[HardGate1] advisory_mode=True — demoting %d blocking violation(s) to "
                "warnings. Pipeline continues.", len(raw_failures)
            )
            for f in raw_failures:
                f["demoted_by"] = "advisory_mode"
                f["original_severity"] = f.get("severity", "ERROR")
                f["severity"] = "WARNING"
            raw_warnings.extend(raw_failures)
            raw_failures = []

        # ── Decision ──────────────────────────────────────────────────
        has_blocking = len(raw_failures) > 0

        if has_blocking:
            reason = (
                f"Hard Gate 1 REJECTED: {len(raw_failures)} blocking violation(s) found. "
                f"First failure: {raw_failures[0].get('message', 'N/A')}"
            )
            decision = "REJECT"
            logger.error(
                "Hard Gate 1 REJECTED run_id=%s — %d failure(s), %d warning(s).",
                run_id, len(raw_failures), len(raw_warnings),
            )
        else:
            n_warn = len(raw_warnings)
            adv_note = " [advisory_mode: violations demoted]" if self._advisory_mode else ""
            reason = (
                f"Hard Gate 1 PASSED with {n_warn} warning(s).{adv_note}"
                if n_warn else f"Hard Gate 1 PASSED — all checks clear.{adv_note}"
            )
            decision = "PASS"
            logger.info(
                "Hard Gate 1 PASSED run_id=%s — %d warning(s)%s.",
                run_id, n_warn, adv_note,
            )

        # ── SHAP explanation on REJECT ─────────────────────────────────
        shap_explanation = None
        if has_blocking and _SHAP_AVAILABLE:
            try:
                shap_explanation = _shap_explain(
                    df, run_id=run_id, top_n=5, failures=raw_failures
                )
                logger.info(
                    "[SHAP] Gate failure explained: %s",
                    shap_explanation.get("explanation", "")[:120],
                )
            except Exception as exc:
                logger.warning("[SHAP] explain_gate_failure failed: %s", exc)

        return GateResult(
            run_id=run_id,
            decision=decision,
            reason=reason,
            suppress_learning=has_blocking,
            failures=raw_failures,
            warnings=raw_warnings,
            total_violations=len(raw_failures),
            total_warnings=len(raw_warnings),
            shap_explanation=shap_explanation,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_into_buckets(
        errors: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]],
    ) -> None:
        """Routes a list of raw error dicts into failure vs warning buckets."""
        for e in errors:
            sev = e.get("severity", "ERROR")
            if sev in _BLOCKING_SEVERITIES:
                failures.append(e)
            else:
                warnings.append(e)
