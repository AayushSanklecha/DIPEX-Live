"""
modeling/model_registry.py
---------------------------
Model versioning and artifact registry.

Stores model artifacts (joblib), eval reports, and metadata under:
  data/model_registry/{run_id}/{model_name}/

Supports: list, load, compare, promote operations.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.modeling.model_registry")


class ModelRegistry:
    """
    File-system backed model registry.

    Usage::

        registry = ModelRegistry("data/model_registry")
        registry.save(run_id, "random_forest", model, eval_report, metadata)
        entry = registry.get_best(metric="roc_auc")
    """

    INDEX_FILE = "registry_index.json"

    def __init__(self, base_dir: str = "data/model_registry") -> None:
        self._base = base_dir
        self._index_path = os.path.join(self._base, self.INDEX_FILE)
        os.makedirs(self._base, exist_ok=True)
        self._index: List[Dict[str, Any]] = self._load_index()

    def _load_index(self) -> List[Dict[str, Any]]:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return []
        return []

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def save(
        self,
        run_id: str,
        model_name: str,
        model: Any,
        eval_report: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persist a model and its metadata. Returns the artifact directory path.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for model serialization: pip install joblib")

        artifact_dir = os.path.join(self._base, run_id, model_name)
        os.makedirs(artifact_dir, exist_ok=True)

        model_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump(model, model_path)

        meta = {
            "run_id": run_id,
            "model_name": model_name,
            "artifact_dir": artifact_dir,
            "model_path": model_path,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "promoted": False,
            "eval_report": eval_report or {},
            **(metadata or {}),
        }

        meta_path = os.path.join(artifact_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._index.append(meta)
        self._save_index()
        logger.info("Model saved: %s / %s → %s", run_id, model_name, artifact_dir)
        return artifact_dir

    def load(self, run_id: str, model_name: str) -> Optional[Any]:
        """Load a model artifact from the registry."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required: pip install joblib")

        path = os.path.join(self._base, run_id, model_name, "model.joblib")
        if not os.path.exists(path):
            logger.warning("Model not found: %s", path)
            return None
        return joblib.load(path)

    def list(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registry entries, optionally filtered by run_id."""
        if run_id:
            return [e for e in self._index if e.get("run_id") == run_id]
        return list(self._index)

    def get_best(self, metric: str = "roc_auc", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """Return the registry entry with the best value for `metric`."""
        candidates = [
            e for e in self._index
            if metric in e.get("eval_report", {}).get("metrics", {})
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda e: e["eval_report"]["metrics"][metric],
            reverse=higher_is_better,
        )[0]

    def promote(self, run_id: str, model_name: str) -> bool:
        """Mark a model as 'promoted' (production champion)."""
        for entry in self._index:
            entry["promoted"] = False  # demote current champion
        for entry in self._index:
            if entry.get("run_id") == run_id and entry.get("model_name") == model_name:
                entry["promoted"] = True
                self._save_index()
                logger.info("Promoted model: %s / %s", run_id, model_name)
                return True
        return False

    def get_promoted(self) -> Optional[Dict[str, Any]]:
        """Return the currently promoted model entry."""
        for entry in self._index:
            if entry.get("promoted"):
                return entry
        return None
