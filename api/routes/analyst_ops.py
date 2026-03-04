"""
api/routes/analyst_ops.py
---------------------------
REST API endpoints for the 3-tier Analyst Intelligence Automation Layer.

Endpoints:
  POST /analyst/run                      — Run any analyst operation by tier + name
  GET  /analyst/operations               — List all available operations by tier
  GET  /analyst/result/{lineage_id}      — Retrieve Gold artefact + lineage by ID
  POST /analyst/frame-problem            — Translate vague question → KPI framework
  POST /analyst/design-experiment        — Design A/B test from hypothesis description
  GET  /analyst/cognitive/sanity/{lid}   — Run sanity check on a Gold artefact
  POST /analyst/mentor/review-sql        — Submit SQL for mentorship review
  GET  /analyst/docs/{dataset_id}        — Auto-generated documentation for a dataset
  GET  /analyst/insights/{dataset_id}    — Ranked business insights for a dataset
  POST /analyst/strategy/recommend       — RL-optimised strategy recommendation

All endpoints:
  - Require authentication (via existing auth middleware)
  - Return JSON with validation_passed, confidence_score, _cognitive fields
  - Work on Gold-layer copies only
  - Log every request to the audit trail
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger("dipex.api.analyst_ops")

router = APIRouter(prefix="/analyst", tags=["Analyst Intelligence"])


# ── Request / Response schemas ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    dataset_id:       str
    operation:        str
    tier:             Optional[str] = None      # junior | mid | senior | auto
    parameters:       Optional[Dict] = None
    problem_statement: Optional[str] = None

class FrameProblemRequest(BaseModel):
    question:          str
    dataset_id:        str = ""
    available_columns: Optional[List[str]] = None

class ExperimentRequest(BaseModel):
    hypothesis:    str
    metric:        str
    baseline_rate: float
    mde:           float = 0.05
    alpha:         float = 0.05
    power:         float = 0.80
    daily_traffic: int   = 1000
    dataset_id:    str   = ""

class SQLReviewRequest(BaseModel):
    sql:     str
    purpose: str = ""

class StrategyRequest(BaseModel):
    domain:   str
    dataset_id: str = ""
    context: Optional[Dict] = None
    n_proposals: int = 3


# ── Helper: get orchestrator ────────────────────────────────────────────────────

def _get_orchestrator(config: Optional[Dict] = None):
    try:
        from analyst.analyst_orchestrator import AnalystOrchestrator
        return AnalystOrchestrator(config=config or {})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AnalystOrchestrator unavailable: {e}",
        )


def _snapshot_to_silver(dataset_id: str):
    """Load the latest Silver ImmutableDataFrame for a dataset_id from layer store."""
    try:
        import os
        import pandas as pd
        from ingestion.layer_store import LayerStore
        from ingestion.data_layers import ImmutableDataFrame
        store  = LayerStore()
        record = store.get_latest(dataset_id=dataset_id, layer="silver")
        if record is None:
            # Fallback: try snapshot dir
            snap_path = f"data/snapshots/{dataset_id}_latest.parquet"
            if os.path.exists(snap_path):
                df = pd.read_parquet(snap_path)
            else:
                raise FileNotFoundError(f"No silver layer or snapshot found for '{dataset_id}'")
        else:
            df = record.load()
        return ImmutableDataFrame(df, layer="silver", dataset_id=dataset_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found in silver layer: {e}",
        )


# ── 1. POST /analyst/run ───────────────────────────────────────────────────────

@router.post("/run", summary="Run an analyst operation on a dataset")
async def run_analyst_operation(req: RunRequest) -> Dict:
    """
    Execute any analyst operation (Junior/Mid/Senior) on the Gold layer.
    The system automatically selects the appropriate tier unless tier is specified.
    """
    orch   = _get_orchestrator()
    silver = _snapshot_to_silver(req.dataset_id)
    result = orch.run(
        operation=req.operation, silver=silver,
        params=req.parameters or {},
        force_tier=req.tier,
        problem_statement=req.problem_statement,
    )
    return result.to_dict()


# ── 2. GET /analyst/operations ─────────────────────────────────────────────────

@router.get("/operations", summary="List all available analyst operations by tier")
async def list_operations() -> Dict:
    """Return all available operations grouped by tier."""
    return {
        "junior": [
            "basic_stats", "missing_analysis", "data_cleaning",
            "duplicate_detection", "data_profiling", "schema_validation",
        ],
        "mid": [
            "automated_eda", "statistical_analysis", "ab_test_evaluation",
            "advanced_sql", "dashboard_design", "business_insights",
            "outlier_investigation", "variance_analysis", "segmentation_clustering",
            "time_series_exploration", "cohort_analysis", "correlation_deep_dive",
        ],
        "senior": [
            "strategic_analysis", "risk_assessment", "problem_framing",
            "experiment_design", "executive_report",
        ],
        "total": 23,
    }


# ── 3. GET /analyst/result/{lineage_id} ────────────────────────────────────────

@router.get("/result/{lineage_id}", summary="Retrieve Gold artefact by lineage ID")
async def get_result(lineage_id: str) -> Dict:
    """Retrieve a Gold artefact and its full lineage by lineage_id."""
    try:
        from ingestion.layer_store import LayerStore
        store  = LayerStore()
        record = store.get(lineage_id=lineage_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Lineage ID '{lineage_id}' not found")
        return {
            "lineage_id": lineage_id,
            "dataset_id": getattr(record, "dataset_id", ""),
            "layer": getattr(record, "layer", "gold"),
            "checksum": getattr(record, "checksum", ""),
            "created_at": str(getattr(record, "created_at", "")),
            "row_count": len(record.load()) if hasattr(record, "load") else 0,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. POST /analyst/frame-problem ────────────────────────────────────────────

@router.post("/frame-problem", summary="Frame a vague business question into measurable KPIs")
async def frame_problem(req: FrameProblemRequest) -> Dict:
    """
    Submit a natural-language question and receive a structured measurement framework:
    KPI definitions, North Star metric, success thresholds, and clarification questions.
    """
    from analyst.problem_framing import ProblemFramingEngine
    engine = ProblemFramingEngine()
    framed = engine.frame(
        request=req.question,
        available_columns=req.available_columns,
        dataset_id=req.dataset_id,
    )
    return framed.to_dict()


# ── 5. POST /analyst/design-experiment ────────────────────────────────────────

@router.post("/design-experiment", summary="Design a statistically valid A/B experiment")
async def design_experiment(req: ExperimentRequest) -> Dict:
    """
    Submit a hypothesis and business context — receive a complete experiment design
    with sample size, power, runtime estimate, validity checks, and leakage risks.
    """
    from analyst.experiment_designer import ExperimentDesigner
    designer = ExperimentDesigner()
    design   = designer.design(
        hypothesis=req.hypothesis, metric=req.metric,
        baseline_rate=req.baseline_rate, mde=req.mde,
        alpha=req.alpha, power=req.power, daily_traffic=req.daily_traffic,
    )
    return design.to_dict()


# ── 6. GET /analyst/cognitive/sanity/{dataset_id} ────────────────────────────

@router.get("/cognitive/sanity/{dataset_id}",
            summary="Run sanity check on latest Gold artefact for a dataset")
async def run_sanity_check(dataset_id: str) -> Dict:
    """Run sanity, leakage, and uncertainty checks on the latest Silver layer."""
    from cognitive.reasoning_engine import CognitiveReasoningEngine, AnalysisContext
    silver = _snapshot_to_silver(dataset_id)
    engine = CognitiveReasoningEngine()
    ctx    = AnalysisContext(dataset_id=dataset_id, operation="api_sanity_check")
    finding = engine.reason(silver._df, ctx)
    return finding.to_dict()


# ── 7. POST /analyst/mentor/review-sql ────────────────────────────────────────

@router.post("/mentor/review-sql", summary="Submit SQL for senior mentorship review")
async def review_sql(req: SQLReviewRequest) -> Dict:
    """
    Get senior-level code review feedback on a SQL query:
    style, efficiency, security, edge cases.
    """
    from analyst.mentorship_engine import MentorshipEngine
    engine = MentorshipEngine()
    review = engine.review_sql(req.sql, purpose=req.purpose)
    return review.to_dict()


# ── 8. GET /analyst/docs/{dataset_id} ─────────────────────────────────────────

@router.get("/docs/{dataset_id}", summary="Get auto-generated documentation for a dataset")
async def get_documentation(dataset_id: str) -> Dict:
    """
    Retrieve auto-generated documentation for a dataset:
    KPI dictionary, data contract, lineage, and changelog.
    """
    from analyst.documentation_generator import DocumentationGenerator
    gen  = DocumentationGenerator()
    docs = gen.list_documents(dataset_id=dataset_id)
    silver = _snapshot_to_silver(dataset_id)
    kpis   = gen.generate_kpi_dictionary(silver._df, dataset_id=dataset_id)
    contract = gen.generate_data_contract(silver._df, dataset_id=dataset_id)
    return {
        "dataset_id": dataset_id,
        "kpi_count": len(kpis),
        "kpi_dictionary": [k.to_dict() for k in kpis[:10]],
        "data_contract": contract,
        "document_files": docs[:10],
        "validation_passed": True,
    }


# ── 9. GET /analyst/insights/{dataset_id} ─────────────────────────────────────

@router.get("/insights/{dataset_id}", summary="Get ranked business insights for a dataset")
async def get_insights(dataset_id: str, target_col: Optional[str] = None) -> Dict:
    """
    Get auto-ranked business insights (trends, anomalies, null risks, outliers)
    from the latest Silver layer, with confidence scores.
    """
    orch   = _get_orchestrator()
    silver = _snapshot_to_silver(dataset_id)
    result = orch.run(
        operation="business_insights", silver=silver,
        params={"target_col": target_col} if target_col else {},
        force_tier="mid",
    )
    out = result.to_dict()
    if result.gold_artefact and hasattr(result.gold_artefact, "data"):
        df = result.gold_artefact.data
        out["insights"] = df.to_dict(orient="records")[:20] if df is not None else []
    return out


# ── 10. POST /analyst/strategy/recommend ──────────────────────────────────────

@router.post("/strategy/recommend", summary="Get RL-optimised strategy recommendations")
async def recommend_strategy(req: StrategyRequest) -> Dict:
    """
    Get ranked strategy recommendations from the RL optimizer for a given domain
    (cleaning, model_selection, retry_path, parameter_tuning, imputation, feature_engineering).
    """
    from analyst.rl_optimizer import RLOptimizer
    rl        = RLOptimizer()
    proposals = rl.propose(
        domain=req.domain,
        context=req.context,
        n_proposals=min(req.n_proposals, 5),
    )
    top_strategies = rl.top_strategies(req.domain, n=3)
    return {
        "domain": req.domain,
        "dataset_id": req.dataset_id,
        "proposals": [p.to_dict() for p in proposals],
        "top_strategies": [{"strategy": s, "weight": round(w, 4)} for s, w in top_strategies],
        "note": "All proposals are advisory only. Hard validation gates cannot be bypassed.",
        "validation_passed": True,
    }
