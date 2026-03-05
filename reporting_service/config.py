"""
reporting_service/config.py
-----------------------------
LLM configuration — HuggingFace Inference API (sole provider).

Set these environment variables (or add to .env):
  HF_API_KEY=hf_your_token_here
  HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2       (optional, this is the default)
  HF_FALLBACK_MODEL=Qwen/Qwen2.5-7B-Instruct             (optional, this is the default)
"""

from __future__ import annotations
import os

# ── Provider ──────────────────────────────────────────────────────────────────
# Locked to HuggingFace. Override via env if needed.
LLM_PROVIDER: str = "huggingface"

# ── Primary model (tried first) ───────────────────────────────────────────────
HF_MODEL_NAME: str = os.environ.get(
    "HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"
)

# ── Fallback model (used automatically if primary fails / is loading) ─────────
HF_FALLBACK_MODEL: str = os.environ.get(
    "HF_FALLBACK_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)

# ── HuggingFace API key ───────────────────────────────────────────────────────
HF_API_KEY: str = os.environ.get("HF_API_KEY", "")

# ── Token limits ──────────────────────────────────────────────────────────────
MAX_TOKENS_PER_REPORT: int = 1500
MAX_PROMPT_TOKENS: int     = 1200

# ── Confidence gating ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.75

# ── Alias kept so old imports don't break ────────────────────────────────────
MODEL_NAME: str = HF_MODEL_NAME
