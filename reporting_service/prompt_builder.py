from __future__ import annotations

"""
Governed prompt builder (STEP 6).

Ensures that the LLM sees ONLY:
  - verified_insights
  - approved_metrics
  - confidence_score
  - qa_passed

NEVER passes raw datasets.
"""

import json
from typing import Any, Dict, List


def build_governed_prompt(
    verified_insights: List[str],
    approved_metrics: Dict[str, Any],
    confidence_score: float,
    qa_passed: bool,
) -> str:
    """
    Builds a strictly governed prompt for the reporting LLM.

    Structure:
      - System Instruction
      - User Content (structured JSON payload)
    """
    system_instruction = (
        "You are a governed analytics reporting engine. "
        "Use ONLY the provided verified insights and approved metrics. "
        "Do NOT invent new facts. Do NOT speculate. "
        "Do NOT introduce any external data. "
        "You MUST explicitly mention the confidence score given to you, "
        "and you MUST explicitly state that validation has passed before this report. "
        "If information is not present in the provided JSON payload, you MUST say "
        "'information not provided' instead of guessing."
    )

    user_payload = {
        "verified_insights": verified_insights,
        "approved_metrics": approved_metrics,
        "confidence_score": confidence_score,
        "qa_passed": bool(qa_passed),
    }

    # The prompt is an instruction followed by a strictly limited JSON payload.
    prompt = (
        "System Instruction:\n"
        f"{system_instruction}\n\n"
        "User Content (JSON):\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "Generate an executive summary, explain the insights, interpret risk, "
        "and translate the findings into business language, STRICTLY based on "
        "the provided JSON content."
    )
    return prompt

