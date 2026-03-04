"""
api/routes/analyst_tiers.py
-----------------------------
Phases 15, 16, 17 — Analyst Tier REST API

Dedicated, production-grade endpoints that expose the three tiers of automated
analyst intelligence via HTTP. Each tier maps directly to its corresponding
analyst module (junior_analyst, mid_analyst, senior_analyst).

All endpoints:
  - Require authenticated request (JWT-protected in app.py)
  - Operate on Gold-layer copies only (enforced within the analyst modules)
  - Return structured JSON with result data + metadata + confidence + lineage
  - Never mutate Bronze or Silver layers

Route groups:
  /api/tiers/junior/*  — Phase 15: Junior analyst operations
  /api/tiers/mid/*     — Phase 16: Mid-level analyst operations
  /api/tiers/senior/*  — Phase 17: Senior analyst operations
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger("dipex.api.analyst_tiers")
router = APIRouter(prefix="/api/tiers", tags=["analyst-tiers"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _make_lm():
    """Create a LayerManager with sensible defaults."""
    from ingestion.data_layers import LayerManager
    config = _load_config()
    return LayerManager.from_config(config)


def _build_silver(data: List[Dict[str, Any]]) -> Any:
    """Convert a list-of-dicts payload into an ImmutableDataFrame."""
    from ingestion.data_layers import ImmutableDataFrame
    df = pd.DataFrame(data)
    return ImmutableDataFrame(df, snapshot_id=f"api_{uuid.uuid4().hex[:8]}")


def _gold_to_response(gold: Any, operation: str) -> Dict[str, Any]:
    """Serialise a GoldArtefact to a JSON-safe response dict."""
    try:
        data_dict = gold.df.head(500).to_dict(orient="records")
    except Exception:
        data_dict = []
    return {
        "operation":       operation,
        "snapshot_id":     getattr(gold, "snapshot_id", None),
        "qa_status":       getattr(gold, "qa_status", "PASS"),
        "confidence_score": getattr(gold, "confidence_score", None),
        "row_count":       len(getattr(gold, "df", pd.DataFrame())),
        "data_preview":    data_dict[:10],
        "metadata":        getattr(gold, "metadata", {}),
    }


# ── Shared request schemas ────────────────────────────────────────────────────

class DataPayload(BaseModel):
    """Inline data rows — use for small datasets in API calls."""
    data: List[Dict[str, Any]] = Field(..., description="List of row dicts")
    snapshot_id: str = Field(default="", description="Optional source snapshot ID")


class SqlRequest(BaseModel):
    data: List[Dict[str, Any]]
    sql: str
    snapshot_id: str = Field(default="")


class AggRequest(BaseModel):
    data: List[Dict[str, Any]]
    group_by: List[str]
    agg_map: Dict[str, str] = Field(..., example={"revenue": "sum", "users": "count"})
    snapshot_id: str = Field(default="")


class PivotRequest(BaseModel):
    data: List[Dict[str, Any]]
    index: List[str]
    columns: str
    values: str
    aggfunc: str = Field(default="sum")
    snapshot_id: str = Field(default="")


class KpiRequest(BaseModel):
    data: List[Dict[str, Any]]
    kpi_definitions: Dict[str, str] = Field(
        ..., example={"RevPerUser": "revenue / users"}
    )
    snapshot_id: str = Field(default="")


class ThresholdRequest(BaseModel):
    data: List[Dict[str, Any]]
    thresholds: Dict[str, List[float]] = Field(
        ..., example={"revenue": [0, 1000000], "age": [0, 120]}
    )
    snapshot_id: str = Field(default="")


class EDARequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: Optional[str] = None
    snapshot_id: str = Field(default="")


class StatRequest(BaseModel):
    data: List[Dict[str, Any]]
    group_col: str
    value_col: str
    alpha: float = Field(default=0.05, ge=0.001, le=0.10)
    snapshot_id: str = Field(default="")


class ABTestRequest(BaseModel):
    data: List[Dict[str, Any]]
    group_col: str
    metric_col: str
    control_group: str = Field(default="control")
    treatment_group: str = Field(default="treatment")
    alpha: float = Field(default=0.05)
    snapshot_id: str = Field(default="")


class ClusterRequest(BaseModel):
    data: List[Dict[str, Any]]
    n_clusters: int = Field(default=4, ge=2, le=20)
    feature_cols: Optional[List[str]] = None
    snapshot_id: str = Field(default="")


class TimeSeriesRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    value_col: str
    snapshot_id: str = Field(default="")


class HypothesisRequest(BaseModel):
    data: List[Dict[str, Any]]
    group_col: str
    value_col: str
    alpha: float = Field(default=0.05)
    snapshot_id: str = Field(default="")


class CohortRequest(BaseModel):
    data: List[Dict[str, Any]]
    cohort_col: str
    time_col: str
    value_col: str
    snapshot_id: str = Field(default="")


class RiskRequest(BaseModel):
    data: List[Dict[str, Any]]
    risk_signals: Dict[str, float] = Field(
        ..., example={"null_rate_col": 0.4, "outlier_col": 0.6}
    )
    snapshot_id: str = Field(default="")


class FeatureEngineeringRequest(BaseModel):
    data: List[Dict[str, Any]]
    numeric_cols: List[str]
    log_cols: Optional[List[str]] = None
    bin_cols: Optional[Dict[str, int]] = None
    interactions: Optional[List[List[str]]] = None
    snapshot_id: str = Field(default="")


class ModelTrainingRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str
    feature_cols: Optional[List[str]] = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    snapshot_id: str = Field(default="")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 15 — JUNIOR ANALYST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/junior/clean", summary="[Phase 15] Basic data cleaning")
async def junior_clean(req: DataPayload):
    """Strip whitespace, normalise null representations, standardise dtypes."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        gold = jr.basic_cleaning(silver, source_snapshot_id=req.snapshot_id)
        return _gold_to_response(gold, "basic_cleaning")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/junior/dedup", summary="[Phase 15] Deduplicate rows")
async def junior_dedup(req: DataPayload):
    """Deduplicate rows using all columns or a specified subset."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        gold = jr.remove_duplicates(silver, source_snapshot_id=req.snapshot_id)
        return _gold_to_response(gold, "remove_duplicates")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/junior/aggregate", summary="[Phase 15] Group-by aggregation")
async def junior_aggregate(req: AggRequest):
    """GROUP BY + aggregation: {'revenue': 'sum', 'users': 'count'}."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        gold = jr.simple_aggregation(
            silver, group_by=req.group_by, agg_map=req.agg_map,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "simple_aggregation")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/junior/pivot", summary="[Phase 15] Pivot table")
async def junior_pivot(req: PivotRequest):
    """Create a pivot table: index × columns values with aggfunc."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        gold = jr.pivot_table(
            silver, index=req.index, columns=req.columns,
            values=req.values, aggfunc=req.aggfunc,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "pivot_table")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/junior/kpi", summary="[Phase 15] KPI tracking via eval expressions")
async def junior_kpi(req: KpiRequest):
    """Compute KPIs from eval expressions: {'RevPerUser': 'revenue / users'}."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        gold = jr.kpi_tracking(
            silver, kpi_definitions=req.kpi_definitions,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "kpi_tracking")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/junior/sql", summary="[Phase 15] SQL query on in-memory data (DuckDB)")
async def junior_sql(req: SqlRequest):
    """Run SQL against in-memory Gold copy using DuckDB."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        gold = jr.sql_query(silver, sql=req.sql, source_snapshot_id=req.snapshot_id)
        return _gold_to_response(gold, "sql_query")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/junior/threshold-check", summary="[Phase 15] Manual threshold bounds check")
async def junior_threshold(req: ThresholdRequest):
    """Flag rows outside defined [min, max] bounds per column."""
    from analyst.junior_analyst import JuniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        jr = JuniorAnalyst(layer_manager=lm)
        thresholds = {c: tuple(v) for c, v in req.thresholds.items()}
        gold = jr.manual_threshold_check(
            silver, thresholds=thresholds, source_snapshot_id=req.snapshot_id
        )
        return _gold_to_response(gold, "manual_threshold_check")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 16 — MID-LEVEL ANALYST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/mid/eda", summary="[Phase 16] Automated EDA")
async def mid_eda(req: EDARequest):
    """Full EDA: distributions, correlations, missingness, outliers, top findings."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.automated_eda(
            silver, target_col=req.target_col,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "automated_eda")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/statistical-analysis", summary="[Phase 16] Statistical hypothesis test")
async def mid_statistical(req: StatRequest):
    """Normality test → t-test or Mann-Whitney → effect size (Cohen's d) → CI."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.statistical_analysis(
            silver, group_col=req.group_col, value_col=req.value_col,
            alpha=req.alpha, source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "statistical_analysis")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/ab-test", summary="[Phase 16] A/B test evaluation")
async def mid_ab_test(req: ABTestRequest):
    """A/B test: uplift, CI, p-value, power, MDE, recommendation."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.ab_test_evaluation(
            silver, group_col=req.group_col, metric_col=req.metric_col,
            control_group=req.control_group, treatment_group=req.treatment_group,
            alpha=req.alpha, source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "ab_test_evaluation")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/sql", summary="[Phase 16] Advanced SQL (CTEs, window functions)")
async def mid_sql(req: SqlRequest):
    """Run analytical SQL with CTEs and window functions on Gold copy."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.advanced_sql(silver, sql=req.sql, source_snapshot_id=req.snapshot_id)
        return _gold_to_response(gold, "advanced_sql")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/business-insights", summary="[Phase 16] Business insight generation")
async def mid_insights(req: EDARequest):
    """Ranked business insights from correlations, distributions, and anomalies."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.business_insights(
            silver, target_col=req.target_col,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "business_insights")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/outliers", summary="[Phase 16] Outlier investigation (IQR or Z-score)")
async def mid_outliers(req: EDARequest):
    """Detect and characterise outliers using IQR or Z-score method."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.outlier_investigation(silver, source_snapshot_id=req.snapshot_id)
        return _gold_to_response(gold, "outlier_investigation")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/clustering", summary="[Phase 16] Segmentation / KMeans clustering")
async def mid_clustering(req: ClusterRequest):
    """KMeans segmentation: adds cluster label column to Gold copy."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.segmentation_clustering(
            silver, n_clusters=req.n_clusters, feature_cols=req.feature_cols,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "segmentation_clustering")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/mid/time-series", summary="[Phase 16] Time-series exploration")
async def mid_time_series(req: TimeSeriesRequest):
    """Rolling stats, trend direction, anomaly flags for a date+value series."""
    from analyst.mid_analyst import MidAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        mid = MidAnalyst(layer_manager=lm)
        gold = mid.time_series_exploration(
            silver, date_col=req.date_col, value_col=req.value_col,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "time_series_exploration")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 17 — SENIOR ANALYST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/senior/hypothesis-test", summary="[Phase 17] Statistical hypothesis test")
async def senior_hypothesis(req: HypothesisRequest):
    """Senior-grade t-test between top-2 groups: p-value, α, reject/fail decision."""
    from analyst.senior_analyst import SeniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        sr = SeniorAnalyst(layer_manager=lm)
        gold = sr.statistical_hypothesis_test(
            silver, group_col=req.group_col, value_col=req.value_col,
            alpha=req.alpha, source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "statistical_hypothesis_test")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/senior/cohort-analysis", summary="[Phase 17] Cohort retention analysis")
async def senior_cohort(req: CohortRequest):
    """Cohort-level mean metrics over time. Useful for retention modeling."""
    from analyst.senior_analyst import SeniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        sr = SeniorAnalyst(layer_manager=lm)
        gold = sr.cohort_analysis(
            silver, cohort_col=req.cohort_col, time_col=req.time_col,
            value_col=req.value_col, source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "cohort_analysis")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/senior/feature-engineering", summary="[Phase 17] Feature engineering")
async def senior_feature_engineering(req: FeatureEngineeringRequest):
    """Log transforms, binning, interaction terms, polynomial features on Gold copy."""
    from analyst.senior_analyst import SeniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        sr = SeniorAnalyst(layer_manager=lm)
        interactions = [tuple(pair) for pair in (req.interactions or [])]
        gold = sr.feature_engineering(
            silver, numeric_cols=req.numeric_cols, log_cols=req.log_cols,
            bin_cols=req.bin_cols, interactions=interactions or None,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "feature_engineering")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/senior/model-train", summary="[Phase 17] Baseline model training + validation")
async def senior_model_train(req: ModelTrainingRequest):
    """Train baseline classifier/regressor on Gold copy. Returns metrics + validation split."""
    from analyst.senior_analyst import SeniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        sr = SeniorAnalyst(layer_manager=lm)
        gold = sr.model_training_validation(
            silver, target_col=req.target_col, feature_cols=req.feature_cols,
            test_size=req.test_size, source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "model_training_validation")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/senior/risk-scoring", summary="[Phase 17] Composite risk scoring")
async def senior_risk(req: RiskRequest):
    """Compute row-level composite risk scores from weighted signals."""
    from analyst.senior_analyst import SeniorAnalyst
    try:
        silver = _build_silver(req.data)
        lm = _make_lm()
        sr = SeniorAnalyst(layer_manager=lm)
        gold = sr.risk_scoring(
            silver, risk_signals=req.risk_signals,
            source_snapshot_id=req.snapshot_id,
        )
        return _gold_to_response(gold, "risk_scoring")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/operations",
    summary="List all analyst tier operations and their metadata",
)
async def list_all_operations():
    """
    Returns the complete catalog of analyst tier operations across all 3 tiers.
    Useful for building dynamic UIs and documentation.
    """
    return {
        "junior": [
            {"operation": "clean",           "route": "/api/tiers/junior/clean",           "description": "Basic data cleaning: whitespace, null normalisation, dtype coercion"},
            {"operation": "dedup",           "route": "/api/tiers/junior/dedup",           "description": "Deduplicate rows"},
            {"operation": "aggregate",       "route": "/api/tiers/junior/aggregate",       "description": "GROUP BY + aggregation"},
            {"operation": "pivot",           "route": "/api/tiers/junior/pivot",           "description": "Pivot table with aggfunc"},
            {"operation": "kpi",             "route": "/api/tiers/junior/kpi",             "description": "KPI tracking via eval expressions"},
            {"operation": "sql",             "route": "/api/tiers/junior/sql",             "description": "DuckDB SQL query on in-memory data"},
            {"operation": "threshold-check", "route": "/api/tiers/junior/threshold-check", "description": "Flag rows outside [min, max] bounds"},
        ],
        "mid": [
            {"operation": "eda",                   "route": "/api/tiers/mid/eda",                   "description": "Full automated EDA"},
            {"operation": "statistical-analysis",  "route": "/api/tiers/mid/statistical-analysis",  "description": "Hypothesis test + effect size + CI"},
            {"operation": "ab-test",               "route": "/api/tiers/mid/ab-test",               "description": "A/B test: uplift, power, MDE"},
            {"operation": "sql",                   "route": "/api/tiers/mid/sql",                   "description": "Advanced SQL: CTEs, window functions"},
            {"operation": "business-insights",     "route": "/api/tiers/mid/business-insights",     "description": "Ranked business insights"},
            {"operation": "outliers",              "route": "/api/tiers/mid/outliers",              "description": "Outlier detection (IQR or Z-score)"},
            {"operation": "clustering",            "route": "/api/tiers/mid/clustering",            "description": "KMeans segmentation"},
            {"operation": "time-series",           "route": "/api/tiers/mid/time-series",           "description": "Rolling stats + trend direction"},
        ],
        "senior": [
            {"operation": "hypothesis-test",      "route": "/api/tiers/senior/hypothesis-test",      "description": "Advanced hypothesis testing"},
            {"operation": "cohort-analysis",      "route": "/api/tiers/senior/cohort-analysis",      "description": "Cohort retention analysis"},
            {"operation": "feature-engineering",  "route": "/api/tiers/senior/feature-engineering",  "description": "Log/bin/interaction/poly transforms"},
            {"operation": "model-train",          "route": "/api/tiers/senior/model-train",          "description": "Baseline model training + metrics"},
            {"operation": "risk-scoring",         "route": "/api/tiers/senior/risk-scoring",         "description": "Composite risk score per row"},
        ],
    }
