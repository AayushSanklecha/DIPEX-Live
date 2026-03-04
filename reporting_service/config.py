from __future__ import annotations

import os

"""
Reporting service configuration (STEP 4).

All model/provider settings are defined here so that future switches
to a hosted provider require changing only this module or setting an env var.

LLM_PROVIDER env var overrides the static value below:
  export LLM_PROVIDER=huggingface   # uses HF Inference API
  export LLM_PROVIDER=local         # uses local Ollama
  export LLM_PROVIDER=openai        # uses OpenAI (placeholder)
"""

# Provider selection: "local" (Ollama) | "huggingface" | "openai" | "fallback"
# Can be overridden at runtime via LLM_PROVIDER env var.
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "local")

# Local model name (Ollama)
MODEL_NAME: str = os.environ.get("OLLAMA_MODEL", "analytics-llm")

# HuggingFace Inference API model (default: Mistral-7B instruction tuned)
HF_MODEL_NAME: str = os.environ.get(
    "HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"
)

# Token limits for prompts and reports
MAX_TOKENS_PER_REPORT: int = 1500
MAX_PROMPT_TOKENS: int = 1200

# Confidence gating
CONFIDENCE_THRESHOLD: float = 0.75

# Cost / usage log file
LOG_FILE: str = "llm_usage_log.json"
