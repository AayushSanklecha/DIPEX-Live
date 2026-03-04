from typing import Dict, Any, List, Optional
import pandas as pd

from learning.experience_memory_v2 import ExperienceMemoryV2

class StrategyOptimizer:
    """Analyzes experience memory to optimize global pipeline strategies."""
    
    def __init__(self, memory: ExperienceMemoryV2):
        self.memory = memory

    def analyze_patterns(self) -> Dict[str, Any]:
        """Examines stored experiences to detect cross-run patterns."""
        # Immutable source-of-record: JSONL experience log
        events = self.memory.list_recent(limit=5000)
        approved = [
            e for e in events if e.get("event_type") == "APPROVED_OUTPUT"
        ]
        if not approved:
            return {"status": "no_data", "message": "Not enough experiences to optimize."}

        rows: List[Dict[str, Any]] = []
        for e in approved:
            payload = e.get("payload", {}) or {}
            ws = payload.get("winning_strategy", {}) or {}
            rows.append(
                {
                    "run_id": e.get("run_id"),
                    "fingerprint": e.get("fingerprint"),
                    "attempt": e.get("attempt"),
                    "confidence_score": payload.get("confidence_score"),
                    "model_type": ws.get("model_type"),
                    "task": ws.get("task"),
                }
            )

        df = pd.DataFrame(rows).dropna(subset=["task", "model_type"])
        if df.empty:
            return {
                "experience_count": len(approved),
                "dominant_strategies": {},
                "system_state": "optimizing",
            }

        dominant = (
            df.groupby("task")["model_type"]
            .agg(lambda x: x.value_counts().index[0])
            .to_dict()
        )

        return {
            "experience_count": len(approved),
            "dominant_strategies": dominant,
            "system_state": "optimizing",
        }
