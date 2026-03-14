"""
validation/qa_gate.py
----------------------
DEPRECATED — Superseded by validation.hard_gate.HardGate.

This legacy class is retained to avoid import breakage in any code that
still references it directly. New code should use HardGate exclusively.

Reasons for deprecation:
  1. QAGate.evaluate() only checks severity == "ERROR" and silently ignores
     "CRITICAL" violations — HardGate blocks on both.
  2. QAGate produces duplicate logging when used alongside HardGate.
  3. QAGate has no concept of the REJECT decision or suppress_learning flag.
  4. HardGate integrates SHAP explainability on rejection.

Migration:
    # Old (DO NOT USE):
    from validation.qa_gate import QAGate
    gate = QAGate()
    ok = gate.evaluate(errors)

    # New:
    from validation.hard_gate import HardGate
    gate = HardGate.from_config(config)
    result = gate.run(df, run_id=run_id)
    ok = result.decision == "PASS"
"""

import logging
import warnings
from typing import Any, Dict, List

logger = logging.getLogger("dipex.qa_gate")


class QAGate:
    """
    DEPRECATED: Use validation.hard_gate.HardGate instead.

    This class is a simplified legacy wrapper that is retained ONLY for backward
    compatibility. It no longer changes the pipeline gate decision.
    """

    def __init__(self, severity_threshold: str = "ERROR") -> None:
        warnings.warn(
            "QAGate is deprecated and will be removed in a future version. "
            "Use validation.hard_gate.HardGate instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.severity_threshold = severity_threshold
        self.logger = logging.getLogger("dipex.qa_gate")

    def evaluate(self, validation_errors: List[Dict[str, Any]]) -> bool:
        """
        DEPRECATED: Delegates to HardGate logic.

        Previously evaluated a flat list of error dicts against an ERROR threshold.
        This path is kept for compatibility ONLY. Prefer HardGate.run().
        """
        critical = [e for e in validation_errors
                    if e.get("severity") in {"ERROR", "CRITICAL"}]

        if critical:
            self.logger.error(
                "QAGate (deprecated): %d blocking violation(s) found. "
                "Migrate to HardGate for full severity handling.",
                len(critical),
            )
            for e in critical:
                self.logger.error("  [%s] %s", e.get("type"), e.get("message"))
            return False

        warnings_list = [e for e in validation_errors if e.get("severity") == "WARNING"]
        if warnings_list:
            self.logger.warning(
                "QAGate (deprecated): %d warning(s). Use HardGate for detailed audit output.",
                len(warnings_list),
            )
        else:
            self.logger.info("QAGate (deprecated): No issues found.")

        return True
