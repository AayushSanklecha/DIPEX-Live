from __future__ import annotations

"""
Output validator (STEP 7).

Enforces:
  - Output must mention the confidence score.
  - Output must mention validation passed.
  - Output must NOT introduce numeric values beyond approved_metrics
    (and confidence_score).
  - Output length must be <= MAX_TOKENS_PER_REPORT.

On violation:
  - Log the error.
  - Allow one retry.
  - If still invalid → raise and reject response.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from .config import MAX_TOKENS_PER_REPORT

logger = logging.getLogger(__name__)

Number = float


def _extract_numbers(text: str) -> List[Number]:
    pattern = re.compile(r"-?\d+(?:\.\d+)?")
    nums: List[Number] = []
    for match in pattern.findall(text):
        try:
            nums.append(float(match))
        except ValueError:
            continue
    return nums


def _allowed_numbers_from_metrics(
    approved_metrics: Dict[str, Any],
    confidence_score: float,
) -> List[Number]:
    allowed: List[Number] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, (int, float)):
            allowed.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(approved_metrics or {})
    allowed.append(float(confidence_score))
    # Also allow percentage form of confidence for reporting
    allowed.append(float(confidence_score * 100.0))
    return allowed


def _numbers_ok(response_text: str, approved_metrics: Dict[str, Any], confidence_score: float) -> bool:
    nums = _extract_numbers(response_text)
    if not nums:
        return True
    allowed = _allowed_numbers_from_metrics(approved_metrics, confidence_score)
    if not allowed:
        # If no approved metrics, then any numeric content is suspicious
        return False
    for n in nums:
        if not any(abs(n - a) <= 1e-6 for a in allowed):
            return False
    return True


def _mentions_confidence(response_text: str, confidence_score: float) -> bool:
    # Require approximate numeric mention of the confidence_score
    cs_str = f"{confidence_score:.2f}"
    return cs_str in response_text


def _mentions_validation_passed(response_text: str) -> bool:
    lower = response_text.lower()
    return ("validation" in lower or "qa" in lower) and any(
        kw in lower for kw in ("passed", "pass", "successful", "validated")
    )


def _within_length(response_text: str) -> bool:
    approx_tokens = len(response_text.split())
    return approx_tokens <= MAX_TOKENS_PER_REPORT


def validate_output(
    response_text: str,
    approved_metrics: Dict[str, Any],
    confidence_score: float,
    qa_passed: bool,
) -> List[str]:
    """
    Returns a list of validation error messages (empty if valid).
    """
    errors: List[str] = []

    if not _mentions_confidence(response_text, confidence_score):
        errors.append("Response does not explicitly mention the confidence score.")

    if qa_passed and not _mentions_validation_passed(response_text):
        errors.append("Response does not explicitly state that validation passed.")

    if not _numbers_ok(response_text, approved_metrics, confidence_score):
        errors.append(
            "Response contains numeric values that are not present in approved metrics "
            "or the confidence score."
        )

    if not _within_length(response_text):
        errors.append(
            f"Response exceeds MAX_TOKENS_PER_REPORT={MAX_TOKENS_PER_REPORT} token limit."
        )

    return errors


def generate_with_validation(
    provider,
    prompt: str,
    approved_metrics: Dict[str, Any],
    confidence_score: float,
    qa_passed: bool,
) -> Tuple[str, int]:
    """
    Wraps provider.generate() with strict validation and one retry.

    Returns:
        (validated_text, tokens_used)

    Raises:
        ValueError if validation fails after one retry.
    """
    last_errors: List[str] = []
    for attempt in (1, 2):
        result = provider.generate(prompt)
        text = str(result.get("text", ""))
        tokens_used = int(result.get("tokens_used", len(text.split())))
        errors = validate_output(text, approved_metrics, confidence_score, qa_passed)
        if not errors:
            return text, tokens_used
        last_errors = errors
        logger.error(
            "LLM output validation failed on attempt %d: %s",
            attempt,
            "; ".join(errors),
        )

    raise ValueError(
        f"LLM output validation failed after retry: {'; '.join(last_errors)}"
    )

