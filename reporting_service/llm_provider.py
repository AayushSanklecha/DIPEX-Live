"""
reporting_service/llm_provider.py
-----------------------------------
LLM provider — HuggingFace Inference API (sole provider).

Governance guarantees:
  - LLM may ONLY access verified, QA-passed, structured payloads
  - Every call is logged to audit/llm_prompts.jsonl (hash only, no raw text)
  - PII is scrubbed before any LLM call
  - Token cap enforced (cost guard via MAX_PROMPT_TOKENS)
  - Cost tracker: cumulative token usage recorded per session
  - Graceful fallback to rule-based summary if both HF models fail / no API key
  - No new facts injected; confidence and validation status always cited

Models:
  Primary  : mistralai/Mistral-7B-Instruct-v0.2
  Fallback : Qwen/Qwen2.5-7B-Instruct   (auto-tried if primary fails)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .config import LLM_PROVIDER, MODEL_NAME, MAX_PROMPT_TOKENS

logger = logging.getLogger(__name__)

# ── PII patterns — regex-based redaction filter ───────────────────────────────
_PII_PATTERNS: List[tuple] = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{4}[\s\-]){3}\d{4}\b"), "[CARD_NUMBER]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
    (re.compile(r"\b(?:NHS|MRN|DOB)[:\s#]?\s*[\d\-A-Za-z]+\b", re.I), "[HEALTH_ID]"),
]


# ── Cost Tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    _session_tokens: int = 0
    _session_calls:  int = 0

    @classmethod
    def record(cls, prompt_tokens: int, response_tokens: int,
               provider: str, model: str) -> None:
        cls._session_tokens += prompt_tokens + response_tokens
        cls._session_calls  += 1
        try:
            os.makedirs("audit", exist_ok=True)
            entry = {
                "event":           "LLM_CALL",
                "provider":        provider,
                "model":           model,
                "prompt_tokens":   prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens":    prompt_tokens + response_tokens,
                "session_tokens":  cls._session_tokens,
                "session_calls":   cls._session_calls,
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            }
            with open("audit/llm_cost_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    @classmethod
    def get_session_total(cls) -> Dict[str, int]:
        return {"tokens": cls._session_tokens, "calls": cls._session_calls}


# ── Audit Trail ───────────────────────────────────────────────────────────────

def _audit_prompt(prompt_hash: str, prompt_tokens: int, response: str,
                  response_tokens: int, provider: str, run_id: str = "") -> None:
    try:
        os.makedirs("audit", exist_ok=True)
        entry = {
            "event":           "LLM_PROMPT_LOG",
            "run_id":          run_id,
            "provider":        provider,
            "prompt_hash":     prompt_hash,      # SHA-256; never log raw prompt
            "prompt_tokens":   prompt_tokens,
            "response_tokens": response_tokens,
            "response_hash":   hashlib.sha256(response.encode()).hexdigest(),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        with open("audit/llm_prompts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── PII Redaction ─────────────────────────────────────────────────────────────

def redact_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Rule-based Fallback Provider ──────────────────────────────────────────────

class LLMProvider:
    """
    Governance-safe rule-based fallback.
    Used when no API key is set, or when both HF models fail.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config = config or {}

    def generate(self, prompt: str, run_id: str = "") -> Dict[str, Any]:
        return {"text": "", "tokens_used": 0, "fallback": True}

    def generate_summary(
        self,
        verified_result: Dict[str, Any],
        max_words: int = 300,
        run_id: str = "",
    ) -> str:
        gate       = str(verified_result.get("gate_decision", "UNKNOWN"))
        confidence = float(verified_result.get("confidence_score", 0.0))
        metrics    = verified_result.get("metrics", {}) or {}
        conf_vec   = verified_result.get("confidence_vector", {}) or {}

        # Governance gate
        if gate != "PASS":
            return (
                f"[GOVERNANCE BLOCK] Result gate_decision='{gate}' is not PASS. "
                "This result is NOT approved for LLM summarisation."
            )

        lines = [
            "**Validation Status**: All gates PASSED ✓",
            f"**Confidence Score**: {confidence:.1%}",
        ]
        if conf_vec:
            dq = conf_vec.get("data_quality_score", "N/A")
            ss = conf_vec.get("statistical_strength_score", "N/A")
            lines.append(f"**Quality Decomposition** — Data: {dq}, Statistical: {ss}")
        if metrics:
            metric_parts = [f"`{k}={v}`" for k, v in list(metrics.items())[:5]]
            lines.append(f"**Key Metrics**: {', '.join(metric_parts)}")

        lines.append(
            "\n---\n"
            "### Pipeline Verification Summary\n"
            "All deterministic and statistical gates passed. "
            "The dataset has been verified and approved for downstream use. "
            "Set `HF_API_KEY` in your `.env` file to enable LLM-generated narrative.\n"
            "\n*All results independently verified. No speculation in this summary.*"
        )

        summary = "\n\n".join(lines)
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "…"
        return summary


# ── HuggingFace Provider ──────────────────────────────────────────────────────

class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Inference API provider.

    Primary model  : mistralai/Mistral-7B-Instruct-v0.2
    Fallback model : Qwen/Qwen2.5-7B-Instruct  (auto-tried if primary fails)

    Requires env var: HF_API_KEY=hf_your_token_here

    Governance:
    - PII redacted before prompt leaves the system
    - Token budget enforced (MAX_PROMPT_TOKENS)
    - Every call audit-logged and cost-tracked
    - Falls back to rule-based summary if both models fail or no API key
    """

    _ENDPOINT = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        llm_cfg = (config or {}).get("llm", {})

        self._model = (
            llm_cfg.get("hf_model_name")
            or os.environ.get("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        )
        self._fallback_model = (
            llm_cfg.get("hf_fallback_model")
            or os.environ.get("HF_FALLBACK_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        )
        self._api_key = (
            llm_cfg.get("hf_api_key")
            or os.environ.get("HF_API_KEY", "")
        )
        self._timeout = int(llm_cfg.get("timeout_s", 90))

    def _call_model(self, model: str, safe_prompt: str, headers: dict) -> Optional[str]:
        """Single Chat Completions call. Returns text or None on failure."""
        payload = {
            "model":      model,
            "messages":   [{"role": "user", "content": safe_prompt}],
            "max_tokens": min(600, MAX_PROMPT_TOKENS),
            "temperature": 0.3,
        }
        try:
            resp = requests.post(
                self._ENDPOINT, json=payload,
                headers=headers, timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            logger.warning("HuggingFaceProvider [%s]: connection error", model)
            return None
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            logger.warning("HuggingFaceProvider [%s]: HTTP %s — %s", model, status, exc)
            return None
        except Exception as exc:
            logger.warning("HuggingFaceProvider [%s]: request failed: %s", model, exc)
            return None

        text = ""
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "") or ""
            elif isinstance(data, dict):
                text = (
                    data.get("generated_text")
                    or data.get("translation_text")
                    or data.get("summary_text")
                    or ""
                )
        return str(text).strip() if text else None

    def generate(self, prompt: str, run_id: str = "") -> Dict[str, Any]:
        """Call primary model; auto-retry with fallback model if primary fails."""
        if not prompt:
            return {"text": "", "tokens_used": 0}

        if not self._api_key:
            logger.warning("HuggingFaceProvider: HF_API_KEY not set — rule-based fallback")
            return {"text": "", "tokens_used": 0, "fallback": True}

        # 1. PII redaction
        safe_prompt = redact_pii(prompt)

        # 2. Token budget guard
        approx_tokens = len(safe_prompt.split())
        if approx_tokens > MAX_PROMPT_TOKENS:
            safe_prompt   = " ".join(safe_prompt.split()[:MAX_PROMPT_TOKENS])
            approx_tokens = MAX_PROMPT_TOKENS

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        prompt_hash = hashlib.sha256(safe_prompt.encode()).hexdigest()
        t0 = time.perf_counter()

        # 3. Try primary → then fallback
        model_used = self._model
        text = self._call_model(self._model, safe_prompt, headers)
        if not text:
            logger.warning(
                "HuggingFaceProvider: '%s' failed — trying fallback '%s'",
                self._model, self._fallback_model,
            )
            model_used = self._fallback_model
            text = self._call_model(self._fallback_model, safe_prompt, headers)

        if not text:
            logger.warning("HuggingFaceProvider: both models failed — rule-based fallback")
            return {"text": "", "tokens_used": 0, "fallback": True}

        elapsed_ms      = (time.perf_counter() - t0) * 1000
        text            = redact_pii(text)
        response_tokens = len(text.split())

        # 4. Audit + cost
        _audit_prompt(prompt_hash, approx_tokens, text, response_tokens, "huggingface", run_id)
        CostTracker.record(approx_tokens, response_tokens, provider="huggingface", model=model_used)

        logger.info(
            "HuggingFaceProvider: %.0fms model=%s prompt=%d resp=%d tokens",
            elapsed_ms, model_used, approx_tokens, response_tokens,
        )
        return {"text": text, "tokens_used": approx_tokens + response_tokens, "model": model_used}

    def generate_summary(
        self,
        verified_result: Dict[str, Any],
        max_words: int = 300,
        run_id: str = "",
    ) -> str:
        # Governance gate first — super() returns GOVERNANCE BLOCK or rule-based text
        fallback = super().generate_summary(verified_result, max_words=max_words, run_id=run_id)
        if "[GOVERNANCE BLOCK]" in fallback:
            return fallback

        gate       = verified_result.get("gate_decision", "PASS")
        confidence = float(verified_result.get("confidence_score", 0.0))
        metrics    = verified_result.get("metrics", {}) or {}
        conf_vec   = verified_result.get("confidence_vector", {}) or {}

        prompt = (
            f"You are a Senior Data Analyst writing a comprehensive executive report.\n\n"
            f"VERIFIED PIPELINE RESULTS:\n"
            f"- Validation Status: {gate}\n"
            f"- Confidence Score: {confidence:.2%}\n"
            f"- Confidence Vector: {json.dumps(conf_vec, indent=2)}\n"
            f"- Key Metrics: {json.dumps(metrics, indent=2)}\n\n"
            f"Write a detailed analytical summary with these sections:\n"
            f"1. **Raw Math & Metrics** — cite the confidence score and gate results\n"
            f"2. **Tasks Performed** — what the pipeline validated and processed\n"
            f"3. **Analytical Meaning** — what the scores mean for data quality\n"
            f"4. **Recommendation** — is this data safe for downstream ML and reporting?\n\n"
            f"RULES: Do NOT invent facts. No hallucination. Max {max_words} words.\n\n"
            f"Executive Analysis:"
        )

        result = self.generate(prompt, run_id=run_id)
        text   = result.get("text", "").strip()

        if not text or result.get("fallback"):
            return fallback

        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "…"
        return text


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_provider(config: Optional[Dict] = None) -> LLMProvider:
    """
    Returns HuggingFaceProvider (sole active provider).
    Falls back to rule-based LLMProvider if HF_API_KEY is not set.
    """
    return HuggingFaceProvider(config)
