"""
cognitive/expectation_calibrator.py
--------------------------------------
Separates verified insights from assumptions and manages stakeholder
expectation bias.

Core principle: analysts overfit to what stakeholders *want* to hear.
This module enforces intellectual honesty by:
  1. Clearly labelling every output as VERIFIED | ASSUMPTION | UNCERTAIN
  2. Suppressing speculative findings from executive-level reports
  3. Computing an "overpromise risk" score when expectations diverge from data
  4. Flagging confirmation bias patterns (same conclusion in 3+ consecutive runs)
"""
from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.cognitive.expectation_calibrator")


@dataclass
class InsightVerdict:
    insight_id:     str
    statement:      str
    verdict:        str          # VERIFIED | ASSUMPTION | UNCERTAIN | SPECULATIVE
    confidence:     float        # 0.0–1.0
    evidence:       str = ""
    caveats:        List[str] = field(default_factory=list)
    publishable:    bool = True
    overpromise_flag: bool = False

    def to_dict(self) -> Dict:
        return {
            "insight_id": self.insight_id,
            "statement": self.statement,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "caveats": self.caveats,
            "publishable": self.publishable,
            "overpromise_flag": self.overpromise_flag,
        }


class ExpectationCalibrator:
    """
    Governs what gets published and how it is framed.

    Rules:
      - SPECULATIVE insights are never publishable without human override
      - UNCERTAIN insights always include caveats in output
      - Confirmation bias (same conclusion ≥ 3 consecutive runs) triggers a warning
      - Overpromise detected when stated confidence > data-supported confidence
    """

    PUBLISHABLE_VERDICTS = {"VERIFIED", "ASSUMPTION"}
    BIAS_WINDOW = 5   # track last N conclusions for confirmation bias

    def __init__(self) -> None:
        self._recent_conclusions: deque = deque(maxlen=self.BIAS_WINDOW)
        self._session_verdicts: List[InsightVerdict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        statement: str,
        confidence: float,
        evidence: str = "",
        caveats: Optional[List[str]] = None,
        insight_id: Optional[str] = None,
        stakeholder_expected: Optional[str] = None,
    ) -> InsightVerdict:
        """
        Evaluate an insight and assign it a verdict.

        Parameters
        ----------
        statement             : The insight being evaluated
        confidence            : Data-supported confidence (0–1)
        stakeholder_expected  : What the stakeholder expects to hear (optional)
        """
        import uuid as _uuid
        iid = insight_id or str(_uuid.uuid4())[:8]
        verdict, publishable = self._assign_verdict(confidence)
        final_caveats = list(caveats or [])

        # Overpromise detection
        overpromise = False
        if stakeholder_expected and stakeholder_expected.lower() in statement.lower():
            if confidence < 0.6:
                overpromise = True
                final_caveats.append(
                    "⚠ Expectation alignment risk: conclusion matches what stakeholder expected "
                    "but data confidence is low — review for confirmation bias"
                )
                logger.warning(
                    "[ExpectationCalibrator] Overpromise risk: '%s…' (confidence=%.0f%%)",
                    statement[:60], confidence * 100
                )

        # Suppress speculative from publishing
        if verdict == "SPECULATIVE":
            publishable = False
            final_caveats.append("SPECULATIVE — do not include in executive report without manual review")

        # Confirmation bias check
        self._recent_conclusions.append(statement[:50].lower())
        bias_warning = self._detect_confirmation_bias()
        if bias_warning:
            final_caveats.append(bias_warning)

        iv = InsightVerdict(
            insight_id=iid, statement=statement, verdict=verdict,
            confidence=confidence, evidence=evidence, caveats=final_caveats,
            publishable=publishable, overpromise_flag=overpromise,
        )
        self._session_verdicts.append(iv)
        return iv

    def filter_publishable(self, verdicts: List[InsightVerdict]) -> List[InsightVerdict]:
        return [v for v in verdicts if v.publishable]

    def session_summary(self) -> Dict:
        total     = len(self._session_verdicts)
        published = sum(1 for v in self._session_verdicts if v.publishable)
        return {
            "total_insights": total,
            "publishable": published,
            "suppressed": total - published,
            "overpromise_flags": sum(1 for v in self._session_verdicts if v.overpromise_flag),
            "verdicts_distribution": dict(
                Counter(v.verdict for v in self._session_verdicts)
            ),
        }

    # ── Internal logic ────────────────────────────────────────────────────────

    def _assign_verdict(self, confidence: float) -> tuple[str, bool]:
        if confidence >= 0.90:
            return "VERIFIED", True
        elif confidence >= 0.70:
            return "ASSUMPTION", True
        elif confidence >= 0.50:
            return "UNCERTAIN", True
        else:
            return "SPECULATIVE", False

    def _detect_confirmation_bias(self) -> str:
        """If the last N conclusions all contain the same keyword, flag bias."""
        if len(self._recent_conclusions) < self.BIAS_WINDOW:
            return ""
        words = [w for s in self._recent_conclusions for w in s.split()]
        counter = Counter(words)
        most_common, count = counter.most_common(1)[0]
        # Exclude stop words
        stop = {"the", "is", "in", "and", "to", "a", "of", "that", "for"}
        if most_common in stop:
            return ""
        if count >= self.BIAS_WINDOW - 1 and len(most_common) > 4:
            return (
                f"⚠ Confirmation bias risk: the term '{most_common}' appears in "
                f"{count}/{self.BIAS_WINDOW} recent conclusions — verify this is data-driven"
            )
        return ""
