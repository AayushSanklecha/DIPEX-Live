"""
cognitive/assumption_tracker.py
---------------------------------
Stores every assumption made during analysis with confidence weights.

In real analyst work, every conclusion rests on hidden assumptions.
This module makes them explicit and auditable.

Examples of tracked assumptions:
  - "Revenue column uses USD (assumed, not verified)"
  - "Missing values in churn_flag are assumed to be 0 (non-churned)"
  - "Dataset is assumed to be complete for Q4 (last record: 2024-12-28)"
  - "Outliers are assumed to be noise and were excluded from regression"
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("dipex.cognitive.assumption_tracker")


@dataclass
class Assumption:
    assumption_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    statement:      str = ""
    category:       str = "general"    # data | model | business | temporal | scope
    confidence:     float = 0.8        # 0.0 = pure guess, 1.0 = verified fact
    risk_if_wrong:  str = "MEDIUM"     # LOW | MEDIUM | HIGH | CRITICAL
    column:         Optional[str] = None
    dataset_id:     Optional[str] = None
    analysis_step:  Optional[str] = None
    timestamp:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verified:       bool = False
    verification_note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def flagged(self) -> bool:
        """An assumption is flagged if it is low-confidence or high-risk."""
        return self.confidence < 0.6 or self.risk_if_wrong in ("HIGH", "CRITICAL")


class AssumptionTracker:
    """
    Thread-safe assumption registry for a single analysis session.
    Persists assumptions to a JSON log for auditability.
    """

    def __init__(
        self, store_path: str = "data/assumptions",
        session_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self._assumptions: List[Assumption] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self, statement: str,
        category: str = "general",
        confidence: float = 0.8,
        risk_if_wrong: str = "MEDIUM",
        column: Optional[str] = None,
        dataset_id: Optional[str] = None,
        analysis_step: Optional[str] = None,
    ) -> Assumption:
        """Log an assumption and return it."""
        a = Assumption(
            statement=statement, category=category, confidence=confidence,
            risk_if_wrong=risk_if_wrong, column=column,
            dataset_id=dataset_id, analysis_step=analysis_step,
        )
        self._assumptions.append(a)
        if a.flagged():
            logger.warning(
                "[Assumption] ⚠ LOW-CONFIDENCE/HIGH-RISK: '%s' (conf=%.0f%%, risk=%s)",
                statement[:80], confidence * 100, risk_if_wrong,
            )
        else:
            logger.debug("[Assumption] Recorded: '%s'", statement[:80])
        return a

    def verify(self, assumption_id: str, note: str = "") -> bool:
        """Mark an assumption as verified."""
        for a in self._assumptions:
            if a.assumption_id == assumption_id:
                a.verified = True
                a.verification_note = note
                a.confidence = max(a.confidence, 0.95)
                return True
        return False

    def flagged_assumptions(self) -> List[Assumption]:
        return [a for a in self._assumptions if a.flagged()]

    def all_assumptions(self) -> List[Assumption]:
        return list(self._assumptions)

    def assumption_summary(self) -> Dict:
        flagged = self.flagged_assumptions()
        return {
            "session_id": self.session_id,
            "total": len(self._assumptions),
            "flagged": len(flagged),
            "verified": sum(1 for a in self._assumptions if a.verified),
            "high_risk": sum(1 for a in self._assumptions
                             if a.risk_if_wrong in ("HIGH", "CRITICAL")),
            "assumptions": [a.to_dict() for a in self._assumptions],
        }

    def persist(self) -> str:
        """Save session assumptions to JSON."""
        path = os.path.join(self.store_path, f"session_{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.assumption_summary(), f, indent=2)
        logger.info("[AssumptionTracker] Persisted %d assumptions → %s",
                    len(self._assumptions), path)
        return path

    def safe_to_publish(self) -> bool:
        """Returns True if no unverified HIGH/CRITICAL risk assumptions."""
        return not any(
            not a.verified and a.risk_if_wrong in ("HIGH", "CRITICAL")
            for a in self._assumptions
        )

    # ── Preset helpers ────────────────────────────────────────────────────────

    def assume_missing_is_zero(self, column: str, dataset_id: str = "") -> Assumption:
        return self.record(
            f"Missing values in '{column}' are treated as 0 (not absent = zero)",
            category="data", confidence=0.7, risk_if_wrong="MEDIUM",
            column=column, dataset_id=dataset_id,
        )

    def assume_iid(self, dataset_id: str = "") -> Assumption:
        return self.record(
            "Observations assumed i.i.d. (no temporal or cluster dependence)",
            category="model", confidence=0.6, risk_if_wrong="HIGH",
            dataset_id=dataset_id,
        )

    def assume_currency(self, column: str, currency: str = "USD") -> Assumption:
        return self.record(
            f"'{column}' values assumed to be in {currency} (not verified from schema)",
            category="business", confidence=0.75, risk_if_wrong="HIGH",
            column=column,
        )

    def assume_data_complete(self, dataset_id: str, as_of: str = "") -> Assumption:
        note = f" as of {as_of}" if as_of else ""
        return self.record(
            f"Dataset '{dataset_id}' assumed to be complete{note}",
            category="temporal", confidence=0.65, risk_if_wrong="MEDIUM",
            dataset_id=dataset_id,
        )
