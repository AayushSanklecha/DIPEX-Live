"""
api/routes/model.py
---------------------
Model training, prediction, and registry endpoints.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/model", tags=["Modeling"])
logger = logging.getLogger("dipex.api.model")


class TrainRequest(BaseModel):
    run_id: str
    target: str
    features: Optional[List[str]] = None
    algorithms: Optional[List[str]] = None


class PredictRequest(BaseModel):
    run_id: str
    model_name: str
    data: List[Dict[str, Any]]        # list of row dicts


@router.post("/train")
async def train(req: TrainRequest):
    """Train ML models for a run_id."""
    import yaml
    from modeling.trainer import ModelTrainer
    from modeling.evaluator import ModelEvaluator
    from modeling.model_registry import ModelRegistry

    data_path = _find_dataset(req.run_id)
    if not data_path:
        raise HTTPException(404, detail=f"Dataset for run_id '{req.run_id}' not found.")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Override algorithms if provided
    if req.algorithms:
        config.setdefault("modeling", {})["algorithms"] = req.algorithms

    df = pd.read_csv(data_path)
    trainer = ModelTrainer.from_config(config)
    results = trainer.train(df, target=req.target, features=req.features, run_id=req.run_id)

    registry = ModelRegistry()
    output = {}
    for algo, result in results.items():
        if result.model is not None:
            # Evaluate
            feats = req.features or [c for c in df.columns if c != req.target]
            sub = df[[req.target] + feats].dropna()
            X = sub[feats].values
            y = sub[req.target].values
            evaluator = ModelEvaluator()
            eval_report = evaluator.evaluate(result.model, X, y, result.task, model_name=algo)

            # Save to registry
            registry.save(req.run_id, algo, result.model, eval_report.to_dict(), result.to_dict())
            output[algo] = {**result.to_dict(), "eval_report": eval_report.to_dict()}
        else:
            output[algo] = result.to_dict()

    # Pick best
    best_algo = max(
        (k for k, v in results.items() if v.model is not None),
        key=lambda k: list(results[k].cv_metrics.values())[0] if results[k].cv_metrics else 0,
        default=None,
    )
    if best_algo:
        registry.promote(req.run_id, best_algo)

    return {
        "run_id": req.run_id,
        "target": req.target,
        "models_trained": list(output.keys()),
        "best_model": best_algo,
        "results": {k: v.get("cv_metrics", {}) for k, v in output.items()},
    }


@router.get("/registry")
async def list_registry():
    """List all models in the registry."""
    from modeling.model_registry import ModelRegistry
    registry = ModelRegistry()
    return {
        "models": registry.list(),
        "promoted": registry.get_promoted(),
    }


@router.post("/predict")
async def predict(req: PredictRequest):
    """Batch predict using a registered model."""
    from modeling.model_registry import ModelRegistry
    registry = ModelRegistry()
    model = registry.load(req.run_id, req.model_name)
    if model is None:
        raise HTTPException(404, detail=f"Model '{req.model_name}' for run_id '{req.run_id}' not found.")

    df = pd.DataFrame(req.data)
    try:
        predictions = model.predict(df).tolist()
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df).tolist()
        return {
            "run_id": req.run_id,
            "model": req.model_name,
            "n_predictions": len(predictions),
            "predictions": predictions,
            "probabilities": proba,
        }
    except Exception as exc:
        raise HTTPException(500, detail=f"Prediction failed: {exc}") from exc


def _find_dataset(run_id: str) -> Optional[str]:
    for suffix in ["_cleaned.csv", "_sample.csv"]:
        path = f"data/uploads/{run_id}{suffix}"
        if os.path.exists(path):
            return path
    return None
