# learning/__init__.py
"""
learning — RL, feedback, and experience memory components for DIPEX.

Exports both ExperienceMemory (v1, ChromaDB-backed, simpler interface) and
ExperienceMemoryV2 (production-grade, HMAC-signed append-only JSONL + ChromaDB).
"""

from learning.experience_memory import ExperienceMemory
from learning.experience_memory_v2 import ExperienceMemoryV2, ExperienceEvent

__all__ = [
    "ExperienceMemory",   # v1: simple ChromaDB store/search interface
    "ExperienceMemoryV2", # v2: HMAC-signed, append-only, production-grade
    "ExperienceEvent",    # immutable event dataclass
]
