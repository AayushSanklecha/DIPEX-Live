"""
qa_control/__init__.py
-----------------------
QA, GOVERNANCE & CONTROL LAYER — Layer 3 of the DIPEX architecture.

Exposes a single QAController that aggregates all 5 sub-components:
  1. Deterministic Validation  (validation/hard_gate.py)
  2. Independent QA Verifiers  (verifier/confidence_vector.py)
  3. Regulatory & Business Rules (validation/regulatory/)
  4. Confidence Scoring        (verifier/confidence_vector.aggregate())
  5. Audit Logs                (audit/audit.jsonl)
"""

from qa_control.controller import QAController, QAResult

__all__ = ["QAController", "QAResult"]
