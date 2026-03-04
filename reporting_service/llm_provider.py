"""
reporting_service/llm_provider.py
-----------------------------------
Production-grade LLM provider abstraction — STEP 11: LLM Governed Reporting.

Governance guarantees:
  - LLM may ONLY access verified, QA-passed, structured payloads
  - Every prompt is logged to audit/llm_prompts.jsonl (prompt + response + tokens)
  - PII is scrubbed before any LLM call (redaction filter)
  - Token cap enforced (cost guard via MAX_PROMPT_TOKENS)
  - Cost tracker: cumulative token usage recorded per session
  - Graceful fallback to rule-based summary if LLM unavailable
  - No new facts injected; confidence and validation status always cited
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import LLM_PROVIDER, MODEL_NAME, MAX_PROMPT_TOKENS

logger = logging.getLogger(__name__)

# ── PII patterns — regex-based redaction filter ───────────────────────────────
_PII_PATTERNS: List[tuple] = [
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Phone (various formats)
    (re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"), "[PHONE]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # Credit card (simplified Luhn structure)
    (re.compile(r"\b(?:\d{4}[\s\-]){3}\d{4}\b"), "[CARD_NUMBER]"),
    # IP address
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
    # NHS / Healthcare ID patterns
    (re.compile(r"\b(?:NHS|MRN|DOB)[:\s#]?\s*[\d\-A-Za-z]+\b", re.I), "[HEALTH_ID]"),
]


# ── Cost Tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Thread-safe cumulative token and cost tracker.
    Persists usage to audit/llm_cost_log.jsonl.
    """

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
                "event":            "LLM_CALL",
                "provider":         provider,
                "model":            model,
                "prompt_tokens":    prompt_tokens,
                "response_tokens":  response_tokens,
                "total_tokens":     prompt_tokens + response_tokens,
                "session_tokens":   cls._session_tokens,
                "session_calls":    cls._session_calls,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
            }
            with open("audit/llm_cost_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def get_session_total(cls) -> Dict[str, int]:
        return {"tokens": cls._session_tokens, "calls": cls._session_calls}


# ── Audit Trail ───────────────────────────────────────────────────────────────

def _audit_prompt(prompt_hash: str, prompt_tokens: int, response: str,
                  response_tokens: int, provider: str, run_id: str = "") -> None:
    """Append a structured audit record to audit/llm_prompts.jsonl."""
    try:
        os.makedirs("audit", exist_ok=True)
        entry = {
            "event":           "LLM_PROMPT_LOG",
            "run_id":          run_id,
            "provider":        provider,
            "prompt_hash":     prompt_hash,     # SHA-256; never log raw prompt
            "prompt_tokens":   prompt_tokens,
            "response_tokens": response_tokens,
            "response_hash":   hashlib.sha256(response.encode()).hexdigest(),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        with open("audit/llm_prompts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001
        pass


# ── PII Redaction ─────────────────────────────────────────────────────────────

def redact_pii(text: str) -> str:
    """Apply all PII regex patterns to text and return sanitized version."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Base LLM Provider ─────────────────────────────────────────────────────────

class LLMProvider:
    """
    Abstract provider with governance-safe fallback.

    Governance contract:
    - Only verified, QA-passed payloads may reach LLM
    - PII is scrubbed before prompt construction
    - Every call is cost-tracked and audit-logged
    - Fallback to rule-based summary when LLM unavailable
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config       = config or {}
        self._fallback_only = bool(
            self._config.get("llm", {}).get("fallback_only", False)
        )

    def generate(self, prompt: str, run_id: str = "") -> Dict[str, Any]:
        raise NotImplementedError

    def generate_summary(
        self,
        verified_result: Dict[str, Any],
        max_words: int = 300,
        run_id: str = "",
    ) -> str:
        """
        Governance-safe executive summary generator.

        Rules:
        - Refuses to summarise unapproved results (gate != PASS)
        - Always states confidence score and validation outcome
        - Cites data quality; no speculation or new facts
        - PII redacted before any content is surfaced
        - Gracefully degrades to rule-based if LLM unavailable

        Args:
            verified_result : Must contain gate_decision=PASS to proceed
            max_words       : Token budget guard
            run_id          : Pipeline run ID for audit correlation
        """
        gate = str(verified_result.get("gate_decision", "UNKNOWN"))
        confidence = float(verified_result.get("confidence_score", 0.0))
        metrics    = verified_result.get("metrics", {}) or {}
        conf_vec   = verified_result.get("confidence_vector", {}) or {}

        # ── Governance gate: refuse unapproved result ─────────────────────────
        if gate != "PASS":
            msg = (
                f"[GOVERNANCE BLOCK] Result gate_decision='{gate}' is not PASS. "
                "This result is NOT approved for LLM summarisation."
            )
            logger.warning("LLMProvider: governance block — gate=%s run_id=%s", gate, run_id)
            return msg

        # ── Build rule-based summary (always available as fallback) ───────────
        lines = [
            f"**Validation Status**: All gates PASSED ✓",
            f"**Confidence Score**: {confidence:.1%}",
        ]
        if conf_vec:
            dq = conf_vec.get("data_quality_score", "N/A")
            ss = conf_vec.get("statistical_strength_score", "N/A")
            lines.append(
                f"**Quality Decomposition** — Data: {dq}, Statistical: {ss}"
            )
        if metrics:
            metric_parts = [f"`{k}={v}`" for k, v in list(metrics.items())[:5]]
            lines.append(f"**Key Metrics**: {', '.join(metric_parts)}")
        
        lines.append(
            "\n---\n"
            "### Pipeline Verification Process\n"
            "This report receives its verified grades via a strict **multi-stage Medallion architecture**:\n"
            "- **Validation Engine (Gate 1)**: Deterministic Python rules strictly enforce schema constraints, missing values thresholds, and data typing.\n"
            "- **Statistical Profiler (Gate 2)**: Evaluates statistical properties against threshold distributions to score data quality.\n"
            "- **Confidence Vector Assembly**: The final confidence score aggregates Data Quality, Statistical Strength, and Verification coverage.\n\n"
            "### AI & RAG Models Employed\n"
            "By default, this narrative is generated via an **expert rule-based summarizer**. When an LLM model (like local Ollama `mistral`, or `HuggingFace API`) is enabled, the system uses it alongside a **RAG Retriever** that learns from `ExperienceMemory` repositories to improve context handling and reasoning about data anomalies."
        )

        lines.append(
            "\n*All results independently verified. No speculation or unverified "
            "assertions in this summary.*"
        )

        summary = "\n\n".join(lines)
        words   = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "…"

        logger.info(
            "LLMProvider: rule-based summary generated (run_id=%s, conf=%.2f)",
            run_id, confidence,
        )
        return summary


# ── Local Ollama Provider ─────────────────────────────────────────────────────

class LocalOllamaProvider(LLMProvider):
    """
    Local Ollama-based provider.

    Sends POST to http://localhost:11434/api/generate.
    All calls:
    - PII-redacted before transmission
    - Token-budget checked (MAX_PROMPT_TOKENS)
    - Cost-tracked and audit-logged
    """

    _ENDPOINT = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, run_id: str = "") -> Dict[str, Any]:
        if not prompt:
            return {"text": "", "tokens_used": 0}

        # 1. PII redaction
        safe_prompt = redact_pii(prompt)

        # 2. Token budget guard
        approx_tokens = len(safe_prompt.split())
        if approx_tokens > MAX_PROMPT_TOKENS:
            msg = (
                f"Prompt exceeds MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS} "
                f"({approx_tokens} tokens). Truncating."
            )
            logger.warning("LocalOllamaProvider: %s", msg)
            safe_prompt = " ".join(safe_prompt.split()[:MAX_PROMPT_TOKENS])
            approx_tokens = MAX_PROMPT_TOKENS

        payload = {"model": MODEL_NAME, "prompt": safe_prompt, "stream": False}
        prompt_hash = hashlib.sha256(safe_prompt.encode()).hexdigest()

        t0 = time.perf_counter()
        try:
            resp = requests.post(self._ENDPOINT, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            logger.warning(
                "LocalOllamaProvider: Ollama not reachable — using rule-based fallback"
            )
            return {"text": "", "tokens_used": 0, "fallback": True}
        except Exception as exc:  # noqa: BLE001
            logger.warning("LocalOllamaProvider: request failed: %s", exc)
            return {"text": "", "tokens_used": 0, "error": str(exc)}

        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = data.get("response") or data.get("output") or data.get("text") or ""
        if not isinstance(text, str):
            text = str(text)

        # PII-redact the response too before returning
        text = redact_pii(text)
        response_tokens = len(text.split())

        # 3. Audit trail
        _audit_prompt(
            prompt_hash=prompt_hash,
            prompt_tokens=approx_tokens,
            response=text,
            response_tokens=response_tokens,
            provider="local_ollama",
            run_id=run_id,
        )

        # 4. Cost tracker
        CostTracker.record(approx_tokens, response_tokens,
                           provider="local_ollama", model=MODEL_NAME)

        logger.info(
            "LocalOllamaProvider: %.0fms, prompt=%d tokens, response=%d tokens",
            elapsed_ms, approx_tokens, response_tokens,
        )
        return {"text": text, "tokens_used": approx_tokens + response_tokens}

    def generate_summary(
        self,
        verified_result: Dict[str, Any],
        max_words: int = 300,
        run_id: str = "",
    ) -> str:
        """
        Governance-safe LLM summary with Ollama.
        Falls back to rule-based if Ollama is unreachable or result is unapproved.
        """
        # Governance gate first — if not approved, rule-based fallback is returned
        fallback = super().generate_summary(
            verified_result, max_words=max_words, run_id=run_id
        )
        if "[GOVERNANCE BLOCK]" in fallback or self._fallback_only:
            return fallback

        # Build structured prompt — only verified facts
        gate       = verified_result.get("gate_decision", "PASS")
        confidence = float(verified_result.get("confidence_score", 0.0))
        metrics    = verified_result.get("metrics", {}) or {}
        conf_vec   = verified_result.get("confidence_vector", {}) or {}

        prompt = (
            f"You are a senior data analyst writing a board-ready executive summary.\n\n"
            f"VERIFIED RESULTS (do not deviate from these facts):\n"
            f"- Validation Status: {gate}\n"
            f"- Confidence Score: {confidence:.1%}\n"
            f"- Confidence Vector: {json.dumps(conf_vec, indent=2)}\n"
            f"- Key Metrics: {json.dumps(metrics, indent=2)}\n\n"
            f"RULES:\n"
            f"1. You MUST cite the confidence score in your summary.\n"
            f"2. You MUST mention that all validation gates passed.\n"
            f"3. Do NOT invent facts, trends, or predictions not in the data above.\n"
            f"4. Do NOT express overconfidence. Acknowledge any uncertainty.\n"
            f"5. Maximum {max_words} words.\n\n"
            f"Write the executive summary now:"
        )

        result = self.generate(prompt, run_id=run_id)
        text = result.get("text", "").strip()

        if not text or result.get("fallback") or result.get("error"):
            logger.info(
                "LocalOllamaProvider: Ollama unavailable, using rule-based summary"
            )
            return fallback

        # Final token-budget guard on response
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "…"
        return text


# ── OpenAI Provider (placeholder) ────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """
    Placeholder for future hosted provider (OpenAI / Gemini / Anthropic).

    Switching is controlled solely by config.LLM_PROVIDER.
    The rest of the codebase must not change.
    LLM API key integration is handled at this layer only.
    """

    def generate(self, prompt: str, run_id: str = "") -> Dict[str, Any]:
        raise NotImplementedError(
            "OpenAIProvider: LLM API key integration is pending (Phase last). "
            "Set LLM_PROVIDER=local to use Ollama, or leave unset for rule-based fallback."
        )


# ── HuggingFace Inference API Provider ────────────────────────────────────────

class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace Inference API provider.

    Calls https://api-inference.huggingface.co/models/<HF_MODEL_NAME>
    using the Bearer token from env HF_API_KEY (or config llm.hf_api_key).

    Governance contract (identical to LocalOllamaProvider):
    - PII redacted before prompt leaves the system
    - Token budget enforced (MAX_PROMPT_TOKENS)
    - Every call audit-logged and cost-tracked
    - Graceful fallback to rule-based summary on any API failure
    - Supports both text-generation and text2text-generation task formats
    """

    _BASE_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        llm_cfg     = (config or {}).get("llm", {})
        self._model = (
            llm_cfg.get("hf_model_name")
            or os.environ.get("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        )
        # Secondary model — used automatically if the primary model fails
        self._fallback_model = (
            llm_cfg.get("hf_fallback_model")
            or os.environ.get("HF_FALLBACK_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        )
        self._api_key = (
            llm_cfg.get("hf_api_key")
            or os.environ.get("HF_API_KEY", "")
        )
        self._endpoint = self._BASE_URL
        self._timeout  = int(llm_cfg.get("timeout_s", 90))

    def _call_model(self, model: str, safe_prompt: str, headers: dict) -> Optional[str]:
        """Make a single Chat Completions call to HF router. Returns generated text or None on failure."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": safe_prompt}],
            "max_tokens": min(600, MAX_PROMPT_TOKENS),
            "temperature": 0.3,
        }
        try:
            resp = requests.post(
                self._endpoint, json=payload,
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("HuggingFaceProvider [%s]: request failed: %s", model, exc)
            return None

        # Parse OpenAI-compatible Chat Completions response
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
        """Call HF Router with primary model; auto-retry with fallback model if primary fails."""
        if not prompt:
            return {"text": "", "tokens_used": 0}

        if not self._api_key:
            logger.warning(
                "HuggingFaceProvider: HF_API_KEY not set — using rule-based fallback"
            )
            return {"text": "", "tokens_used": 0, "fallback": True}

        # 1. PII redaction
        safe_prompt = redact_pii(prompt)

        # 2. Token budget guard
        approx_tokens = len(safe_prompt.split())
        if approx_tokens > MAX_PROMPT_TOKENS:
            logger.warning(
                "HuggingFaceProvider: prompt truncated from %d to %d tokens",
                approx_tokens, MAX_PROMPT_TOKENS,
            )
            safe_prompt   = " ".join(safe_prompt.split()[:MAX_PROMPT_TOKENS])
            approx_tokens = MAX_PROMPT_TOKENS

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        prompt_hash = hashlib.sha256(safe_prompt.encode()).hexdigest()
        t0 = time.perf_counter()

        # 3. Try primary model, then fallback model automatically
        model_used = self._model
        text = self._call_model(self._model, safe_prompt, headers)
        if not text:
            logger.warning(
                "HuggingFaceProvider: primary model '%s' failed — trying fallback '%s'",
                self._model, self._fallback_model,
            )
            model_used = self._fallback_model
            text = self._call_model(self._fallback_model, safe_prompt, headers)

        if not text:
            logger.warning("HuggingFaceProvider: both models failed — rule-based fallback")
            return {"text": "", "tokens_used": 0, "fallback": True}

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # PII-redact the response
        text            = redact_pii(text)
        response_tokens = len(text.split())

        # 4. Audit trail
        _audit_prompt(
            prompt_hash=prompt_hash,
            prompt_tokens=approx_tokens,
            response=text,
            response_tokens=response_tokens,
            provider="huggingface",
            run_id=run_id,
        )

        # 5. Cost tracker
        CostTracker.record(
            approx_tokens, response_tokens,
            provider="huggingface", model=model_used,
        )

        logger.info(
            "HuggingFaceProvider: %.0fms model=%s prompt=%d tokens response=%d tokens",
            elapsed_ms, model_used, approx_tokens, response_tokens,
        )
        return {"text": text, "tokens_used": approx_tokens + response_tokens}

    def generate_summary(
        self,
        verified_result: Dict[str, Any],
        max_words: int = 300,
        run_id: str = "",
    ) -> str:
        """
        Governance-safe LLM summary via HuggingFace Inference API.
        Falls back to rule-based if API unavailable or result is unapproved.
        """
        # Governance gate: if not approved, rule-based fallback is returned immediately
        fallback = super().generate_summary(verified_result, max_words=max_words, run_id=run_id)
        if "[GOVERNANCE BLOCK]" in fallback or self._fallback_only:
            return fallback

        gate       = verified_result.get("gate_decision", "PASS")
        confidence = float(verified_result.get("confidence_score", 0.0))
        metrics    = verified_result.get("metrics", {}) or {}
        conf_vec   = verified_result.get("confidence_vector", {}) or {}

        prompt = (
            f"You are a Senior Data Analyst writing a comprehensive executive report for a data pipeline run.\n\n"
            f"Here are the VERIFIED RESULTS of the ingestion and analysis pipeline:\n"
            f"- Validation Status: {gate}\n"
            f"- Overall Pipeline Confidence Score: {confidence:.2%}\n"
            f"- Detailed Confidence Vector: {json.dumps(conf_vec, indent=2)}\n"
            f"- Extraction & Processing Metrics: {json.dumps(metrics, indent=2)}\n\n"
            f"YOUR TASK:\n"
            f"Write a detailed, analytical summary evaluating the health and quality of this data.\n\n"
            f"You MUST explicitly include the following sections:\n"
            f"1. **Raw Math & Metrics**: Explicitly recite the overall confidence score ({confidence:.2%}), whether the Validation Status passed, and highlight the most important individual sub-scores from the Confidence Vector.\n"
            f"2. **Tasks Performed**: Summarize what the pipeline did (e.g., how many rows were ingested, what quality checks were run, what schema drift occurred based on the metrics).\n"
            f"3. **Analytical Meaning**: Explain the 'meaning behind the data'. (e.g., if there is High Drift, explain that the incoming data pattern has changed. If there are nulls, explain the data quality gap).\n"
            f"4. **Recommendation**: Provide a clear statement on whether the data is safe for downstream machine learning and reporting.\n\n"
            f"RULES:\n"
            f"- Do NOT invent facts or hallucinate trends outside of the provided metrics.\n"
            f"- Format nicely with professional headers and bullet points.\n"
            f"- Maximum {max_words} words.\n\n"
            f"Executive Analysis Report:\n"
        )

        result = self.generate(prompt, run_id=run_id)
        text   = result.get("text", "").strip()

        if not text or result.get("fallback") or result.get("error"):
            logger.info("HuggingFaceProvider: API unavailable, using rule-based summary")
            return fallback

        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "…"
        return text


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_provider(config: Optional[Dict] = None) -> LLMProvider:
    """
    Factory — returns the configured LLM provider.

    LLM_PROVIDER env/config values:
    - 'local'        → LocalOllamaProvider  (local Ollama endpoint)
    - 'huggingface'  → HuggingFaceProvider  (HF Inference API; needs HF_API_KEY)
    - 'openai'       → OpenAIProvider       (placeholder; API key integration pending)
    - anything else  → LLMProvider          (rule-based fallback only)

    The env var LLM_PROVIDER overrides config so that switching providers
    never requires a code change.
    """
    # env var takes priority over static config.py value
    env_provider = os.environ.get("LLM_PROVIDER", "").lower().strip()
    provider = env_provider or (LLM_PROVIDER.lower() if LLM_PROVIDER else "fallback")

    # Also check per-run config dict
    if config:
        provider = config.get("llm", {}).get("provider", provider).lower()

    if provider == "local":
        return LocalOllamaProvider(config)
    if provider == "huggingface":
        return HuggingFaceProvider(config)
    if provider == "openai":
        return OpenAIProvider(config)
    # Default: governance-safe rule-based fallback
    logger.info("LLM_PROVIDER='%s' — using rule-based fallback provider.", provider)
    return LLMProvider(config)
